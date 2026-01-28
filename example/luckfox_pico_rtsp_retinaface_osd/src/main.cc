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
#include <sys/time.h>
#include <vector>
#include <algorithm>  // REQUIRED for std::max and std::min

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "retinaface.h"

// OpenCV includes for AI processing
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

// Hardware alignment and AI model constants
#define RK_ALIGN_16(x) (((x) + 15) & (~15))
#define MODEL_WIDTH  640
#define MODEL_HEIGHT 640

// RTSP input URL
#define RTSP_INPUT_URL "rtsp://220.254.72.200/Src/MediaInput/h264/stream_2"

// Global flag for graceful shutdown
static volatile bool g_running = true;

void signal_handler(int sig) {
    printf("\nReceived signal %d, initiating graceful shutdown...\n", sig);
    g_running = false;
}

uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char *argv[]) {
    system("RkLunch-stop.sh");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    RK_S32 s32Ret = 0;
    int ret = 0;
    
    // *** AI Model variables - DECLARE AT TOP ***
    rknn_app_context_t rknn_app_ctx;
    object_detect_result_list od_results;
    const char *model_path = "./model/retinaface.rknn";
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    // *** ALL buffer and matrix pointers - DECLARE AT TOP ***
    unsigned char *temp_nv12_buffer = NULL;
    cv::Mat *bgr_frame = NULL;
    cv::Mat *model_bgr = NULL;
    
    // VENC stream structure
    VENC_STREAM_S stFrame;
    memset(&stFrame, 0, sizeof(stFrame));
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    if (!stFrame.pstPack) {
        fprintf(stderr, "ERROR: Failed to allocate VENC_PACK_S\n");
        return -1;
    }
    stFrame.u32PackCount = 1;
    
    RK_U32 H264_TimeRef = 0;
    
    // FFmpeg variables
    AVFormatContext *formatContext = NULL;
    AVCodecContext *codecContext = NULL;
    AVCodec *codec = NULL;
    AVPacket packet;
    AVFrame *avframe = NULL;
    AVFrame *nv12_frame = NULL;
    struct SwsContext *sws_ctx = NULL;
    int videoStreamIndex = -1;
    
    // Stream dimensions
    int width = 0, height = 0, vir_width = 0, vir_height = 0, uv_height = 0;
    
    // Rockchip MPI resources
    MB_POOL src_Pool = MB_INVALID_POOLID;
    MB_BLK src_Blk = MB_INVALID_HANDLE;
    unsigned char *data = NULL;
    
    // RTSP handles
    rtsp_demo_handle g_rtsplive = NULL;
    rtsp_session_handle g_rtsp_session = NULL;
    
    // Performance tracking
    uint64_t frame_count = 0, start_time = 0, last_fps_time = 0, fps_frame_count = 0;
    int consecutive_errors = 0;

    // AI scaling factors
    float scale_x = 0.0f;
    float scale_y = 0.0f;
    
    printf("========================================\n");
    printf("RTSP Stream Relay with AI Inference\n");
    printf("========================================\n");
    
    // Initialize AI Model
    if (init_retinaface_model(model_path, &rknn_app_ctx) != RK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to initialize RetinaFace model\n");
        ret = -1;
        return -1;
    }
    printf("✓ RetinaFace AI model initialized\n");
    
    // FFmpeg initialization
    avformat_network_init();
    
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    
    printf("Connecting to: %s\n", RTSP_INPUT_URL);
    
    if (avformat_open_input(&formatContext, RTSP_INPUT_URL, NULL, &opts) != 0) {
        fprintf(stderr, "ERROR: Failed to open RTSP stream\n");
        av_dict_free(&opts);
        ret = -1;
        return -1;
    }
    av_dict_free(&opts);
    printf("✓ Connected successfully\n");
    
    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to find stream information\n");
        ret = -1;
        return -1;
    }
    printf("✓ Stream analysis complete\n");
    
    // Find video stream
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            width = formatContext->streams[i]->codecpar->width;
            height = formatContext->streams[i]->codecpar->height;
            
            vir_width = RK_ALIGN_16(width);
            vir_height = height;
            uv_height = (height + 1) / 2;
            
            printf("✓ Video stream: %dx%d (stride: %d)\n", width, height, vir_width);
            break;
        }
    }
    
    if (videoStreamIndex == -1) {
        fprintf(stderr, "ERROR: No video stream found\n");
        ret = -1;
        return -1;
    }
    
    // Set AI scaling factors
    scale_x = (float)width / (float)MODEL_WIDTH;
    scale_y = (float)height / (float)MODEL_HEIGHT;
    
    // Decoder initialization
    codecContext = avcodec_alloc_context3(NULL);
    if (!codecContext) {
        ret = -1;
        return -1;
    }
    
    avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar);
    codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec || avcodec_open2(codecContext, codec, NULL) < 0) {
        fprintf(stderr, "ERROR: Decoder initialization failed\n");
        ret = -1;
        return -1;
    }
    printf("✓ Video decoder initialized\n");
    
    // Swscale for NV12
    sws_ctx = sws_getContext(
		codecContext->width, codecContext->height, codecContext->pix_fmt,
		width, height, AV_PIX_FMT_YUV420P,  // I420
		SWS_FAST_BILINEAR, NULL, NULL, NULL);
    
    if (!sws_ctx) {
        ret = -1;
        return -1;
    }
    printf("✓ Format converter initialized (NV12 output)\n");
    
    // Frame allocation
    avframe = av_frame_alloc();
    nv12_frame = av_frame_alloc();
    if (!avframe || !nv12_frame) {
        ret = -1;
        return -1;
    }
    
    nv12_frame->format = AV_PIX_FMT_YUV420P;
    nv12_frame->width = width;
    nv12_frame->height = height;
    if (av_frame_get_buffer(nv12_frame, 32) < 0) {
		fprintf(stderr, "ERROR: Failed to allocate I420 frame buffer\n");
        ret = -1;
        return -1;
    }
    
    av_init_packet(&packet);
    
    // Rockchip MPI initialization
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        ret = -1;
        return -1;
    }
    printf("✓ Rockchip MPI system initialized\n");
    
    // Memory pool
    MB_POOL_CONFIG_S PoolCfg;
    memset(&PoolCfg, 0, sizeof(PoolCfg));
    PoolCfg.u64MBSize = vir_width * vir_height * 3 / 2;
    PoolCfg.u32MBCnt = 3;
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
    
    src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    if (src_Pool == MB_INVALID_POOLID) {
        ret = -1;
        return -1;
    }
    
    src_Blk = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_TRUE);
    if (src_Blk == MB_INVALID_HANDLE) {
        ret = -1;
        return -1;
    }
    
    data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    if (!data) {
        ret = -1;
        return -1;
    }
    printf("✓ Memory pool created (%lu bytes, %d buffers)\n", 
           PoolCfg.u64MBSize, PoolCfg.u32MBCnt);
    
    // VENC frame configuration
    VIDEO_FRAME_INFO_S h264_frame;
    memset(&h264_frame, 0, sizeof(h264_frame));
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = vir_width;
    h264_frame.stVFrame.u32VirHeight = vir_height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_YUV420P;   // I420
    h264_frame.stVFrame.u32FrameFlag = 160;
    h264_frame.stVFrame.pMbBlk = src_Blk;
    
    // RTSP server
    g_rtsplive = create_rtsp_demo(554);
    if (!g_rtsplive) {
        ret = -1;
        return -1;
    }
    
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsp_session) {
        ret = -1;
        return -1;
    }
    
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("✓ RTSP server initialized on port 554\n");
    
    // VENC initialization
    RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
    s32Ret = venc_init(0, width, height, enCodecType);
    if (s32Ret != RK_SUCCESS) {
        ret = -1;
        return -1;
    }
    
    VENC_RECV_PIC_PARAM_S stRecvParam;
    memset(&stRecvParam, 0, sizeof(stRecvParam));
    stRecvParam.s32RecvPicNum = -1;
    s32Ret = RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);
    if (s32Ret != RK_SUCCESS) {
        ret = -1;
        return -1;
    }
    printf("✓ Video encoder channel started\n");
    
    // *** Allocate AI processing buffers AFTER dimensions are known ***
    temp_nv12_buffer = (unsigned char *)malloc(width * height * 3 / 2);
    if (!temp_nv12_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate temporary NV12 buffer\n");
        ret = -1;
        return -1;
    }
    
    bgr_frame = new cv::Mat(height, width, CV_8UC3);
    model_bgr = new cv::Mat(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
    
    printf("\n========================================\n");
    printf("Starting AI-enhanced frame processing...\n");
    printf("Press Ctrl+C to stop gracefully\n");
    printf("========================================\n\n");
    
    start_time = get_time_us();
    last_fps_time = start_time;
    
    // Main processing loop with AI integration
    while (g_running && av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                while (avcodec_receive_frame(codecContext, avframe) == 0) {
                    if (!data) continue;
                    
                    // Step 1: Convert FFmpeg frame to NV12
                    sws_scale(sws_ctx,
                              (const uint8_t * const*)avframe->data, 
                              avframe->linesize,
                              0, codecContext->height,
                              nv12_frame->data, nv12_frame->linesize);
                    
                    /// Step 2: Copy I420 from FFmpeg to aligned VENC buffer
					// I420 layout: Y plane (full size), U plane (1/4 size), V plane (1/4 size)

					// Copy Y plane (luminance)
					for (int i = 0; i < height; i++) {
						memcpy(data + i * vir_width, 
							nv12_frame->data[0] + i * nv12_frame->linesize[0], 
							width);
					}

					// Copy U plane (chrominance)
					int u_offset = vir_width * vir_height;
					for (int i = 0; i < height / 2; i++) {
						memcpy(data + u_offset + i * (vir_width / 2), 
							nv12_frame->data[1] + i * nv12_frame->linesize[1],  // data[1] = U plane
							width / 2);
					}

					// Copy V plane (chrominance)
					int v_offset = u_offset + (vir_width * vir_height / 4);
					for (int i = 0; i < height / 2; i++) {
						memcpy(data + v_offset + i * (vir_width / 2), 
							nv12_frame->data[2] + i * nv12_frame->linesize[2],  // data[2] = V plane
							width / 2);
					}
                    
                    // *** AI PROCESSING SECTION - COMPLETELY CORRECTED ***

					// Step 3: Create stride-corrected I420 data for OpenCV
					// Copy Y plane (remove stride padding)
					for (int i = 0; i < height; i++) {
						memcpy(temp_nv12_buffer + i * width,
							data + i * vir_width,
							width);
					}

					// Copy U plane (remove stride padding)
					int temp_u_offset = width * height;
					int src_u_offset = vir_width * vir_height;
					for (int i = 0; i < height / 2; i++) {
						memcpy(temp_nv12_buffer + temp_u_offset + i * (width / 2),
							data + src_u_offset + i * (vir_width / 2),
							width / 2);
					}

					// Copy V plane (remove stride padding)
					int temp_v_offset = temp_u_offset + (width * height / 4);
					int src_v_offset = src_u_offset + (vir_width * vir_height / 4);
					for (int i = 0; i < height / 2; i++) {
						memcpy(temp_nv12_buffer + temp_v_offset + i * (width / 2),
							data + src_v_offset + i * (vir_width / 2),
							width / 2);
					}

					// Step 4: Convert I420 to BGR for AI processing (CORRECTED DIRECTION)
					cv::Mat i420_mat(height + height/2, width, CV_8UC1, temp_nv12_buffer);
					cv::cvtColor(i420_mat, *bgr_frame, cv::COLOR_YUV2BGR_I420);  // ✅ Correct: I420 → BGR

					// Step 5: Prepare AI model input
					cv::resize(*bgr_frame, *model_bgr, cv::Size(MODEL_WIDTH, MODEL_HEIGHT), 
							0, 0, cv::INTER_LINEAR);

					// Step 6: Run AI inference
					memcpy(rknn_app_ctx.input_mems[0]->virt_addr, model_bgr->data, 
						MODEL_WIDTH * MODEL_HEIGHT * 3);
					inference_retinaface_model(&rknn_app_ctx, &od_results);

					// Step 7: Draw detection results
					for (int i = 0; i < od_results.count; i++) {
						object_detect_result *det_result = &(od_results.results[i]);
						
						int sX = (int)((float)det_result->box.left * scale_x);
						int sY = (int)((float)det_result->box.top * scale_y);
						int eX = (int)((float)det_result->box.right * scale_x);
						int eY = (int)((float)det_result->box.bottom * scale_y);
						
						// Clamp coordinates to frame bounds
						sX = std::max(0, std::min(sX, width-1));
						sY = std::max(0, std::min(sY, height-1));
						eX = std::max(0, std::min(eX, width-1));
						eY = std::max(0, std::min(eY, height-1));
						
						cv::rectangle(*bgr_frame, cv::Point(sX, sY), cv::Point(eX, eY), 
									cv::Scalar(0, 255, 0), 3);
					}

					// Step 8: Convert annotated BGR back to I420 (CORRECT DIRECTION)
					cv::cvtColor(*bgr_frame, i420_mat, cv::COLOR_BGR2YUV_I420);  // ✅ Correct: BGR → I420

					// Step 9: Copy processed I420 back to VENC buffer with stride alignment
					// Copy Y plane
					for (int i = 0; i < height; i++) {
						memcpy(data + i * vir_width,
							temp_nv12_buffer + i * width,
							width);
					}

					// Copy U plane (reuse offsets from Step 3)
					int dst_u_offset = vir_width * vir_height;
					for (int i = 0; i < height / 2; i++) {
						memcpy(data + dst_u_offset + i * (vir_width / 2),
							temp_nv12_buffer + temp_u_offset + i * (width / 2),
							width / 2);
					}

					// Copy V plane (reuse offsets from Step 3)
					int dst_v_offset = dst_u_offset + (vir_width * vir_height / 4);
					for (int i = 0; i < height / 2; i++) {
						memcpy(data + dst_v_offset + i * (vir_width / 2),
							temp_nv12_buffer + temp_v_offset + i * (width / 2),
							width / 2);
					}
					// *** END AI PROCESSING SECTION ***
                    
                    // Step 10: Hardware encoding
                    RK_MPI_SYS_MmzFlushCache(src_Blk, RK_FALSE);
                    
                    h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
                    h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
                    
                    s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
                    if (s32Ret != RK_SUCCESS) {
                        printf("WARNING: VENC_SendFrame failed: %#x\n", s32Ret);
                        consecutive_errors++;
                        if (consecutive_errors > 10) {
                            fprintf(stderr, "ERROR: Too many consecutive encoder errors\n");
                            g_running = false;
                            break;
                        }
                        continue;
                    }
                    consecutive_errors = 0;
                    
                    // Get encoded stream and transmit via RTSP
                    s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, 1000);
                    if (s32Ret == RK_SUCCESS) {
                        if (g_rtsplive && g_rtsp_session) {
                            void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                            rtsp_tx_video(g_rtsp_session, 
                                          (uint8_t *)pData, 
                                          stFrame.pstPack->u32Len,
                                          stFrame.pstPack->u64PTS);
                            rtsp_do_event(g_rtsplive);
                        }
                        RK_MPI_VENC_ReleaseStream(0, &stFrame);
                    } else if (s32Ret != RK_ERR_VENC_BUF_EMPTY) {
                        printf("WARNING: VENC_GetStream failed: %#x\n", s32Ret);
                    }
                    
                    // Performance tracking with AI stats
                    frame_count++;
                    fps_frame_count++;
                    
                    uint64_t current_time = get_time_us();
                    uint64_t elapsed_us = current_time - last_fps_time;
                    
                    if (elapsed_us >= 1000000) {  // Every second
                        double fps = (double)fps_frame_count / (elapsed_us / 1000000.0);
                        double total_elapsed = (current_time - start_time) / 1000000.0;
                        
                        printf("[%.1fs] Frames: %lu | FPS: %.2f | Faces: %d\n",
                               total_elapsed,
                               (unsigned long)frame_count,
                               fps,
                               od_results.count);
                        
                        fps_frame_count = 0;
                        last_fps_time = current_time;
                    }
                }
            }
        }
        av_packet_unref(&packet);
    }
    
    // Final statistics
    uint64_t end_time = get_time_us();
    double total_time = (end_time - start_time) / 1000000.0;
    double avg_fps = total_time > 0 ? (double)frame_count / total_time : 0;
    
    printf("\n========================================\n");
    printf("AI Processing Complete\n");
    printf("========================================\n");
    printf("Final Statistics:\n");
    printf("  - Total frames processed: %lu\n", (unsigned long)frame_count);
    printf("  - Total processing time: %.2f seconds\n", total_time);
    printf("  - Average FPS: %.2f\n", avg_fps);
    printf("========================================\n");


    printf("\nInitiating resource cleanup...\n");
    
    // *** FREE OPENCV MATRICES FIRST ***
    if (model_bgr) {
        delete model_bgr;
        printf("✓ Model BGR matrix freed\n");
    }
    
    if (bgr_frame) {
        delete bgr_frame;
        printf("✓ BGR frame matrix freed\n");
    }
    
    // Free temporary AI buffer
    if (temp_nv12_buffer) {
        free(temp_nv12_buffer);
        printf("✓ Temporary NV12 buffer freed\n");
    }
    
    if (stFrame.pstPack) {
        free(stFrame.pstPack);
        printf("✓ VENC pack structure freed\n");
    }
    
    // FFmpeg cleanup
    if (nv12_frame) {
        av_frame_free(&nv12_frame);
        printf("✓ NV12 frame buffer freed\n");
    }
    
    if (avframe) {
        av_frame_free(&avframe);
        printf("✓ Input frame buffer freed\n");
    }
    
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        printf("✓ Format converter freed\n");
    }
    
    if (codecContext) {
        avcodec_free_context(&codecContext);
        printf("✓ Decoder context freed\n");
    }
    
    if (formatContext) {
        avformat_close_input(&formatContext);
        printf("✓ Input stream closed\n");
    }
    
    avformat_network_deinit();
    printf("✓ FFmpeg network deinitialized\n");
    
    // Rockchip MPI cleanup
    if (src_Blk != MB_INVALID_HANDLE) {
        RK_MPI_MB_ReleaseMB(src_Blk);
        printf("✓ Memory block released\n");
    }
    
    if (src_Pool != MB_INVALID_POOLID) {
        RK_MPI_MB_DestroyPool(src_Pool);
        printf("✓ Memory pool destroyed\n");
    }
    
    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);
    printf("✓ Video encoder stopped\n");
    
    if (g_rtsplive) {
        rtsp_del_demo(g_rtsplive);
        printf("✓ RTSP server stopped\n");
    }
    
    RK_MPI_SYS_Exit();
    printf("✓ Rockchip MPI system exited\n");
    
    // Release AI model
    release_retinaface_model(&rknn_app_ctx);
    printf("✓ RetinaFace model released\n");
    
    printf("\nCleanup complete. Exiting.\n");
    
    return ret;

}