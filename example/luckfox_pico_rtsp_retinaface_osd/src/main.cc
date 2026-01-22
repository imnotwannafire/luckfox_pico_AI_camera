#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "retinaface.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

#define VENC_CHN 0
#define MODEL_WIDTH  640
#define MODEL_HEIGHT 640
#define AI_PROCESS_INTERVAL 3  // Process every 3rd frame for performance

// Global variables
MPP_CHN_S stvencChn;
VENC_RECV_PIC_PARAM_S stRecvParam;

rknn_app_context_t rknn_app_ctx;	
object_detect_result_list od_results;

rtsp_demo_handle g_rtsplive = NULL;
rtsp_session_handle g_rtsp_session;

static volatile bool g_venc_ready = false;

// Control flags
static volatile bool g_should_exit = false;

// RTSP input configuration - Updated with your URL
static char g_rtsp_url[256] = "rtsp://220.254.72.200/Src/MediaInput/h264/stream_2";
static int g_stream_width = 0;
static int g_stream_height = 0;

// Signal handler for graceful shutdown
static void signal_handler(int signo) {
    printf("Received signal %d, preparing to exit...\n", signo);
    g_should_exit = true;
}

// Memory cleanup callback for MPI buffers - must return int
static int user_data_callback(void *opaque) {
    if (opaque) {
        free(opaque);
    }
    return 0;
}

// Process AI detection on a frame
static void process_ai_detection(AVFrame *frame, AVCodecContext *codec_ctx, struct SwsContext *sws_ctx) {
    // Allocate BGR frame for conversion
    AVFrame *bgr_frame = av_frame_alloc();
    if (!bgr_frame) {
        printf("ERROR: Failed to allocate BGR frame\n");
        return;
    }
    
    bgr_frame->format = AV_PIX_FMT_BGR24;
    bgr_frame->width = codec_ctx->width;
    bgr_frame->height = codec_ctx->height;
    
    if (av_frame_get_buffer(bgr_frame, 0) < 0) {
        printf("ERROR: Failed to allocate BGR frame buffer\n");
        av_frame_free(&bgr_frame);
        return;
    }
    
    // Convert YUV to BGR using swscale
    sws_scale(sws_ctx, 
              (const uint8_t * const*)frame->data, frame->linesize, 
              0, codec_ctx->height,
              bgr_frame->data, bgr_frame->linesize);
    
    // Create OpenCV Mat from BGR frame
    cv::Mat bgr(codec_ctx->height, codec_ctx->width, CV_8UC3, 
                bgr_frame->data[0], bgr_frame->linesize[0]);
    
    // Resize for model input
    cv::Mat model_bgr(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
    cv::resize(bgr, model_bgr, cv::Size(MODEL_WIDTH, MODEL_HEIGHT), 0, 0, cv::INTER_LINEAR);
    
    // Copy data to RKNN input buffer
    if (rknn_app_ctx.input_mems && rknn_app_ctx.input_mems[0]) {
        memcpy(rknn_app_ctx.input_mems[0]->virt_addr, model_bgr.data, 
               MODEL_WIDTH * MODEL_HEIGHT * 3);
        
        // Run inference
        int ret = inference_retinaface_model(&rknn_app_ctx, &od_results);
        if (ret == 0) {
            // Process detection results
            float scale_x = (float)codec_ctx->width / (float)MODEL_WIDTH;
            float scale_y = (float)codec_ctx->height / (float)MODEL_HEIGHT;
            
            if (od_results.count > 0) {
                printf("Detected %d faces:\n", od_results.count);
                for (int i = 0; i < od_results.count; i++) {
                    object_detect_result *det_result = &(od_results.results[i]);
                    int left = (int)(det_result->box.left * scale_x);
                    int top = (int)(det_result->box.top * scale_y);
                    int right = (int)(det_result->box.right * scale_x);
                    int bottom = (int)(det_result->box.bottom * scale_y);
                    printf("  Face %d: [%d, %d, %d, %d] confidence: %.2f\n", 
                           i, left, top, right, bottom, det_result->prop);
                }
            }
        }
    }
    
    av_frame_free(&bgr_frame);
}

// Send frame to VENC for encoding
static int send_frame_to_venc(AVFrame *frame, AVCodecContext *codec_ctx, uint64_t frame_index) {
    // Calculate buffer size for NV12 format
    size_t y_size = codec_ctx->width * codec_ctx->height;
    size_t uv_size = y_size / 2;
    size_t total_size = y_size + uv_size;

	// Wait for VENC to be ready
    if (!g_venc_ready) {
        return 0;  // Skip silently until VENC is ready
    }
    
    // Allocate buffer for YUV data
    RK_U8 *yuv_data = (RK_U8 *)malloc(total_size);
    if (!yuv_data) {
        printf("ERROR: Failed to allocate YUV buffer\n");
        return -1;
    }
    
    // Copy YUV data from AVFrame to buffer
    // Handle different pixel formats
    if (codec_ctx->pix_fmt == AV_PIX_FMT_YUV420P || 
        codec_ctx->pix_fmt == AV_PIX_FMT_YUVJ420P) {
        // YUV420P: Y, U, V are in separate planes, need to convert to NV12
        uint8_t *dst_y = yuv_data;
        uint8_t *dst_uv = yuv_data + y_size;
        
        // Copy Y plane
        for (int i = 0; i < codec_ctx->height; i++) {
            memcpy(dst_y + i * codec_ctx->width, 
                   frame->data[0] + i * frame->linesize[0], 
                   codec_ctx->width);
        }
        
        // Interleave U and V planes to create UV plane (NV12)
        for (int i = 0; i < codec_ctx->height / 2; i++) {
            for (int j = 0; j < codec_ctx->width / 2; j++) {
                dst_uv[i * codec_ctx->width + j * 2] = 
                    frame->data[1][i * frame->linesize[1] + j];  // U
                dst_uv[i * codec_ctx->width + j * 2 + 1] = 
                    frame->data[2][i * frame->linesize[2] + j];  // V
            }
        }
    } else if (codec_ctx->pix_fmt == AV_PIX_FMT_NV12) {
        // NV12: Already in correct format
        // Copy Y plane
        for (int i = 0; i < codec_ctx->height; i++) {
            memcpy(yuv_data + i * codec_ctx->width, 
                   frame->data[0] + i * frame->linesize[0], 
                   codec_ctx->width);
        }
        // Copy UV plane
        for (int i = 0; i < codec_ctx->height / 2; i++) {
            memcpy(yuv_data + y_size + i * codec_ctx->width, 
                   frame->data[1] + i * frame->linesize[1], 
                   codec_ctx->width);
        }
    } else {
        printf("ERROR: Unsupported pixel format: %d\n", codec_ctx->pix_fmt);
        free(yuv_data);
        return -1;
    }
    
    // Create MB_BLK with external buffer
    MB_BLK mb_blk = RK_NULL;
    MB_EXT_CONFIG_S ext_config;
    memset(&ext_config, 0, sizeof(MB_EXT_CONFIG_S));
    ext_config.pu8VirAddr = yuv_data;
    ext_config.u64Size = total_size;
    ext_config.pFreeCB = user_data_callback;
    ext_config.pOpaque = yuv_data;
    
    int ret = RK_MPI_SYS_CreateMB(&mb_blk, &ext_config);
    if (ret != RK_SUCCESS) {
        printf("ERROR: RK_MPI_SYS_CreateMB failed: %#x\n", ret);
        free(yuv_data);
        return -1;
    }
    
    // Setup VIDEO_FRAME_INFO_S
    VIDEO_FRAME_INFO_S venc_frame;
    memset(&venc_frame, 0, sizeof(VIDEO_FRAME_INFO_S));
    venc_frame.stVFrame.pMbBlk = mb_blk;
    venc_frame.stVFrame.u32Width = codec_ctx->width;
    venc_frame.stVFrame.u32Height = codec_ctx->height;
    venc_frame.stVFrame.u32VirWidth = codec_ctx->width;
    venc_frame.stVFrame.u32VirHeight = codec_ctx->height;
    venc_frame.stVFrame.enPixelFormat = RK_FMT_YUV420SP;  // NV12
    
    // Calculate PTS based on frame rate (assuming 25fps)
    venc_frame.stVFrame.u64PTS = frame_index * 40000;  // 40ms per frame for 25fps
    venc_frame.stVFrame.u32TimeRef = frame_index;
    venc_frame.stVFrame.u32FrameFlag = 0;
    
    // Send frame to VENC
    ret = RK_MPI_VENC_SendFrame(VENC_CHN, &venc_frame, -1);
    if (ret != RK_SUCCESS) {
        printf("ERROR: RK_MPI_VENC_SendFrame failed: %#x\n", ret);
        RK_MPI_MB_ReleaseMB(mb_blk);
        return -1;
    }
    
    // Release MB_BLK (callback will free yuv_data)
    RK_MPI_MB_ReleaseMB(mb_blk);
    return 0;
}

// Cleanup and reinitialize RTSP connection
static int handle_reconnect(AVFormatContext **fmt_ctx, AVCodecContext **codec_ctx, 
                           struct SwsContext **sws_ctx, int *video_stream_idx) {
    printf("Attempting to reconnect to RTSP stream...\n");
    
    // Cleanup existing resources
    if (*codec_ctx) {
        avcodec_free_context(codec_ctx);
        *codec_ctx = NULL;
    }
    if (*sws_ctx) {
        sws_freeContext(*sws_ctx);
        *sws_ctx = NULL;
    }
    if (*fmt_ctx) {
        avformat_close_input(fmt_ctx);
        *fmt_ctx = NULL;
    }
    
    // Wait before reconnecting
    sleep(2);
    
    // Setup RTSP options
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);  // Use TCP for reliability
    av_dict_set(&opts, "max_delay", "500000", 0);    // 500ms max delay
    av_dict_set(&opts, "stimeout", "5000000", 0);    // 5 second socket timeout
    av_dict_set(&opts, "buffer_size", "1024000", 0); // 1MB buffer
    
    // Open RTSP stream
    printf("Connecting to: %s\n", g_rtsp_url);
    if (avformat_open_input(fmt_ctx, g_rtsp_url, NULL, &opts) < 0) {
        printf("ERROR: Failed to reconnect to RTSP stream\n");
        av_dict_free(&opts);
        return -1;
    }
    av_dict_free(&opts);
    
    // Find stream info
    printf("Finding stream info...\n");
    if (avformat_find_stream_info(*fmt_ctx, NULL) < 0) {
        printf("ERROR: Failed to find stream info\n");
        avformat_close_input(fmt_ctx);
        return -1;
    }
    
    // Find video stream
    *video_stream_idx = -1;
    for (unsigned int i = 0; i < (*fmt_ctx)->nb_streams; i++) {
        if ((*fmt_ctx)->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            *video_stream_idx = i;
            printf("Found video stream at index %d\n", i);
            printf("  Codec: %s\n", avcodec_get_name((*fmt_ctx)->streams[i]->codecpar->codec_id));
            printf("  Size: %dx%d\n", 
                   (*fmt_ctx)->streams[i]->codecpar->width,
                   (*fmt_ctx)->streams[i]->codecpar->height);
            break;
        }
    }
    
    if (*video_stream_idx == -1) {
        printf("ERROR: No video stream found\n");
        avformat_close_input(fmt_ctx);
        return -1;
    }
    
    // Find and open decoder
    const AVCodec *codec = avcodec_find_decoder((*fmt_ctx)->streams[*video_stream_idx]->codecpar->codec_id);
    if (!codec) {
        printf("ERROR: Failed to find codec\n");
        avformat_close_input(fmt_ctx);
        return -1;
    }
    
    *codec_ctx = avcodec_alloc_context3(codec);
    if (!*codec_ctx) {
        printf("ERROR: Failed to allocate codec context\n");
        avformat_close_input(fmt_ctx);
        return -1;
    }
    
    if (avcodec_parameters_to_context(*codec_ctx, (*fmt_ctx)->streams[*video_stream_idx]->codecpar) < 0) {
        printf("ERROR: Failed to copy codec parameters\n");
        avcodec_free_context(codec_ctx);
        avformat_close_input(fmt_ctx);
        return -1;
    }
    (*codec_ctx)->pix_fmt = AV_PIX_FMT_YUV420P;
    if (avcodec_open2(*codec_ctx, codec, NULL) < 0) {
        printf("ERROR: Failed to open codec\n");
        avcodec_free_context(codec_ctx);
        avformat_close_input(fmt_ctx);
        return -1;
    }
    
    // Create scaling context for YUV to BGR conversion
    *sws_ctx = sws_getContext((*codec_ctx)->width, (*codec_ctx)->height, (*codec_ctx)->pix_fmt,
                              (*codec_ctx)->width, (*codec_ctx)->height, AV_PIX_FMT_BGR24,
                              SWS_BILINEAR, NULL, NULL, NULL);
    if (!*sws_ctx) {
        printf("ERROR: Failed to create swscale context\n");
        avcodec_free_context(codec_ctx);
        avformat_close_input(fmt_ctx);
        return -1;
    }
    
    printf("Successfully connected to RTSP stream: %dx%d\n", 
           (*codec_ctx)->width, (*codec_ctx)->height);
    
    return 0;
}

// RTSP decoder and AI processor thread
static void *RtspDecoderAndProcessor(void *arg) {
    printf("======== Starting RTSP Decoder Thread ========\n");
    
    AVFormatContext *fmt_ctx = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVPacket pkt;
    AVFrame *frame = av_frame_alloc();
    int video_stream_idx = -1;
    struct SwsContext *sws_ctx = NULL;
    uint64_t frame_count = 0;
    int ai_frame_counter = 0;
    
    if (!frame) {
        printf("ERROR: Failed to allocate AVFrame\n");
        return NULL;
    }
    
    // Initialize FFmpeg network
    avformat_network_init();
    
    // Initial connection
    if (handle_reconnect(&fmt_ctx, &codec_ctx, &sws_ctx, &video_stream_idx) < 0) {
        printf("ERROR: Failed to initialize RTSP connection\n");
        av_frame_free(&frame);
        avformat_network_deinit();
        return NULL;
    }
    
    // Store stream dimensions for VENC validation
    g_stream_width = codec_ctx->width;
    g_stream_height = codec_ctx->height;
    
    printf("Starting frame processing loop...\n");
    
    // Main processing loop
    while (!g_should_exit) {
        int ret = av_read_frame(fmt_ctx, &pkt);
        
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                printf("RTSP stream ended\n");
            } else {
                char errbuf[128];
                av_strerror(ret, errbuf, sizeof(errbuf));
                printf("ERROR: av_read_frame failed: %s\n", errbuf);
            }
            
            // Attempt reconnection
            if (handle_reconnect(&fmt_ctx, &codec_ctx, &sws_ctx, &video_stream_idx) < 0) {
                printf("Reconnection failed, will retry in 5 seconds...\n");
                sleep(5);
                continue;
            }
            // Update dimensions after reconnect
            g_stream_width = codec_ctx->width;
            g_stream_height = codec_ctx->height;
            frame_count = 0;
            ai_frame_counter = 0;
            continue;
        }
        
        // Process only video packets
        if (pkt.stream_index != video_stream_idx) {
            av_packet_unref(&pkt);
            continue;
        }
        
        // Send packet to decoder
        ret = avcodec_send_packet(codec_ctx, &pkt);
        if (ret < 0) {
            printf("ERROR: avcodec_send_packet failed\n");
            av_packet_unref(&pkt);
            continue;
        }
        
        // Receive decoded frames
		int receive_ret = 0;
        while ((receive_ret = avcodec_receive_frame(codec_ctx, frame)) == 0) {
			// Run AI detection on every Nth frame for performance
			if (++ai_frame_counter >= AI_PROCESS_INTERVAL) {
				process_ai_detection(frame, codec_ctx, sws_ctx);
				ai_frame_counter = 0;
			}
			
			// Always send frame to VENC for encoding and streaming
			if (send_frame_to_venc(frame, codec_ctx, frame_count) < 0) {
				static int error_count = 0;
				if (++error_count % 100 == 0) {
					printf("ERROR: Failed to send frame to VENC (count: %d)\n", error_count);
				}
			}
			
			frame_count++;
		}

		// Print EVERY frame for debugging
		printf("Frame %lu processed, receive_ret=%d\n", frame_count, receive_ret);

		// Only log receive errors if not EAGAIN (which is normal)
		if (receive_ret != AVERROR(EAGAIN)) {
			printf("WARNING: avcodec_receive_frame error: %d\n", receive_ret);
		}

		av_packet_unref(&pkt);
    }
    
    printf("Exiting RTSP decoder thread...\n");
    
    // Cleanup resources
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (frame) av_frame_free(&frame);
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
    avformat_network_deinit();
    
    return NULL;
}

// VENC output and RTSP streaming thread
static void *GetMediaBuffer(void *arg) {
    (void)arg;
    printf("======== Starting VENC Output Thread ========\n");
    
    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    if (!stFrame.pstPack) {
        printf("ERROR: Failed to allocate VENC_PACK_S\n");
        return NULL;
    }
    
    int frame_count = 0;
    
    while (!g_should_exit) {
        int ret = RK_MPI_VENC_GetStream(VENC_CHN, &stFrame, 1000);  // 1 second timeout
        
        if (ret == RK_SUCCESS) {
            // Send encoded frame to RTSP server
            if (g_rtsplive && g_rtsp_session) {
                void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                if (pData) {
                    rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len,
                                  stFrame.pstPack->u64PTS);
                    rtsp_do_event(g_rtsplive);
                    
                    frame_count++;
                    if (frame_count % 100 == 0) {
                        printf("Streamed %d encoded frames\n", frame_count);
                    }
                }
            }
            
            // Release stream
            ret = RK_MPI_VENC_ReleaseStream(VENC_CHN, &stFrame);
            if (ret != RK_SUCCESS) {
                printf("ERROR: RK_MPI_VENC_ReleaseStream failed: %#x\n", ret);
            }
        } else if (ret != RK_ERR_VENC_BUF_EMPTY) {
            // Only log if not timeout/empty buffer
            // printf("WARNING: RK_MPI_VENC_GetStream failed: %#x\n", ret);
        }
        
        usleep(1000);  // Small sleep to prevent busy wait
    }
    
    printf("Exiting VENC output thread...\n");
    free(stFrame.pstPack);
    return NULL;
}

// Main function
int main(int argc, char *argv[]) {
    int ret = 0;
    
    // Parse command line arguments
    if (argc > 1) {
        strncpy(g_rtsp_url, argv[1], sizeof(g_rtsp_url) - 1);
        g_rtsp_url[sizeof(g_rtsp_url) - 1] = '\0';
    }
    
    printf("========================================\n");
    printf("RTSP AI Detection Application\n");
    printf("Input:  %s\n", g_rtsp_url);
    printf("Output: rtsp://[luckfox_ip]:554/live/0\n");
    printf("========================================\n");
    
    // Stop any existing RK processes
    system("RkLunch-stop.sh");
    usleep(500000);  // Wait 500ms
    
    // Setup signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize RKNN model
    const char *model_path = "./model/retinaface.rknn";
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    printf("Loading RKNN model: %s\n", model_path);
    if (init_retinaface_model(model_path, &rknn_app_ctx) != RK_SUCCESS) {
        printf("ERROR: Failed to initialize RKNN model\n");
        return -1;
    }
    printf("RKNN model loaded successfully\n");
    
    // Initialize RK MPI system
    printf("Initializing RK MPI system...\n");
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        printf("ERROR: RK MPI system init failed\n");
        release_retinaface_model(&rknn_app_ctx);
        return -1;
    }
    printf("RK MPI system initialized\n");
    
    // Initialize RTSP server
    printf("Starting RTSP server on port 554...\n");
    g_rtsplive = create_rtsp_demo(554);
    if (!g_rtsplive) {
        printf("ERROR: Failed to create RTSP server\n");
        RK_MPI_SYS_Exit();
        release_retinaface_model(&rknn_app_ctx);
        return -1;
    }
    
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsp_session) {
        printf("ERROR: Failed to create RTSP session\n");
        rtsp_del_demo(g_rtsplive);
        RK_MPI_SYS_Exit();
        release_retinaface_model(&rknn_app_ctx);
        return -1;
    }
    
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("RTSP server started successfully\n");
    
    // Start decoder thread first to get stream dimensions
    printf("Starting RTSP decoder thread...\n");
    pthread_t decoder_thread;
    if (pthread_create(&decoder_thread, NULL, RtspDecoderAndProcessor, NULL) != 0) {
        printf("ERROR: Failed to create decoder thread\n");
        rtsp_del_session(g_rtsp_session);
        rtsp_del_demo(g_rtsplive);
        RK_MPI_SYS_Exit();
        release_retinaface_model(&rknn_app_ctx);
        return -1;
    }
    
    // Wait for first frame to get stream dimensions
    printf("Waiting for stream dimensions...\n");
    int wait_count = 0;
    while (g_stream_width == 0 && g_stream_height == 0 && wait_count < 100) {
        usleep(100000);  // 100ms
        wait_count++;
        if (wait_count % 10 == 0) {
            printf("  Still waiting... (%d/100)\n", wait_count);
        }
    }
    
    if (g_stream_width == 0 || g_stream_height == 0) {
        printf("WARNING: Failed to get stream dimensions in time, using default 720x480\n");
        g_stream_width = 720;
        g_stream_height = 480;
    }
    
    printf("Stream dimensions: %dx%d\n", g_stream_width, g_stream_height);
    
    // Initialize VENC with actual stream dimensions using the existing function
    printf("Initializing VENC...\n");
    RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
    if (venc_init(VENC_CHN, g_stream_width, g_stream_height, enCodecType) != RK_SUCCESS) {
        printf("ERROR: VENC initialization failed\n");
        g_should_exit = true;
        pthread_join(decoder_thread, NULL);
        rtsp_del_session(g_rtsp_session);
        rtsp_del_demo(g_rtsplive);
        RK_MPI_SYS_Exit();
        release_retinaface_model(&rknn_app_ctx);
        return -1;
    }
    
    // Setup VENC channel info
    stvencChn.enModId = RK_ID_VENC;
    stvencChn.s32DevId = 0;
    stvencChn.s32ChnId = VENC_CHN;
    
    printf("VENC initialized successfully\n");
	g_venc_ready = true;
    
    // Start VENC output thread
    printf("Starting VENC output thread...\n");
    pthread_t venc_thread;
    if (pthread_create(&venc_thread, NULL, GetMediaBuffer, NULL) != 0) {
        printf("ERROR: Failed to create VENC thread\n");
        g_should_exit = true;
        pthread_join(decoder_thread, NULL);
        RK_MPI_VENC_StopRecvFrame(VENC_CHN);
        RK_MPI_VENC_DestroyChn(VENC_CHN);
        rtsp_del_session(g_rtsp_session);
        rtsp_del_demo(g_rtsplive);
        RK_MPI_SYS_Exit();
        release_retinaface_model(&rknn_app_ctx);
        return -1;
    }
    
    printf("\n========================================\n");
    printf("Application started successfully!\n");
    printf("Input:  rtsp://220.254.72.200/Src/MediaInput/h264/stream_2\n");
    printf("Output: rtsp://[luckfox_ip]:554/live/0\n");
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");
    
    // Main loop - just wait for exit signal
    while (!g_should_exit) {
        sleep(1);
    }
    
    printf("\n========================================\n");
    printf("Shutting down gracefully...\n");
    printf("========================================\n");
    
    // Wait for threads to finish
    printf("Waiting for decoder thread...\n");
    pthread_join(decoder_thread, NULL);
    
    printf("Waiting for VENC thread...\n");
    pthread_join(venc_thread, NULL);
    
    // Cleanup resources
    printf("Stopping VENC...\n");
    RK_MPI_VENC_StopRecvFrame(VENC_CHN);
    RK_MPI_VENC_DestroyChn(VENC_CHN);
    
    printf("Stopping RTSP server...\n");
    if (g_rtsp_session) {
        rtsp_del_session(g_rtsp_session);
    }
    if (g_rtsplive) {
        rtsp_del_demo(g_rtsplive);
    }
    
    printf("Shutting down RK MPI...\n");
    RK_MPI_SYS_Exit();
    
    printf("Releasing RKNN model...\n");
    release_retinaface_model(&rknn_app_ctx);
    
    printf("========================================\n");
    printf("Application exited cleanly\n");
    printf("========================================\n");
    
    return 0;
}