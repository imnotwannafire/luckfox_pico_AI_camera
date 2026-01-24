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

#include "rtsp_demo.h"
#include "luckfox_mpi.h"

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

// Alignment macro for Rockchip hardware (16-byte boundary requirement)
#define RK_ALIGN_16(x) (((x) + 15) & (~15))

// RTSP input URL
#define RTSP_INPUT_URL "rtsp://220.254.72.200/Src/MediaInput/h264/stream_2"

// Global flag for graceful shutdown
static volatile bool g_running = true;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, initiating graceful shutdown...\n", sig);
    g_running = false;
}

// Get current time in microseconds
uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char *argv[]) {
    system("RkLunch-stop.sh");
    
    // Setup signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    RK_S32 s32Ret = 0;
    int ret = 0;
    
    // VENC stream structure - properly initialized
    // CORRECT:
	VENC_STREAM_S stFrame;
	memset(&stFrame, 0, sizeof(stFrame));
	stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
	if (!stFrame.pstPack) {
		fprintf(stderr, "ERROR: Failed to allocate VENC_PACK_S\n");
		ret = -1;
		return -1;
	}
    
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
    
    // Stream dimensions and alignment
    int width = 0;
    int height = 0;
    int vir_width = 0;   // 16-byte aligned width
    int vir_height = 0;
    int uv_height = 0;   // UV plane height (height/2 for NV12)
    
    // Rockchip MPI resources
    MB_POOL src_Pool = MB_INVALID_POOLID;
    MB_BLK src_Blk = MB_INVALID_HANDLE;
    unsigned char *data = NULL;
    
    // RTSP server handles
    rtsp_demo_handle g_rtsplive = NULL;
    rtsp_session_handle g_rtsp_session = NULL;
    
    // Performance tracking
    uint64_t frame_count = 0;
    uint64_t start_time = 0;
    uint64_t last_fps_time = 0;
    uint64_t fps_frame_count = 0;
    int consecutive_errors = 0;
    
    printf("========================================\n");
    printf("RTSP Stream Relay - NV12 Optimized\n");
    printf("========================================\n");
    
    // Initialize FFmpeg network subsystem
    avformat_network_init();
    
    // Configure RTSP connection options
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);        // 500ms max delay
    av_dict_set(&opts, "stimeout", "5000000", 0);        // 5s connection timeout
    av_dict_set(&opts, "buffer_size", "1024000", 0);     // 1MB buffer
    
    printf("Connecting to: %s\n", RTSP_INPUT_URL);
    
    if (avformat_open_input(&formatContext, RTSP_INPUT_URL, NULL, &opts) != 0) {
        fprintf(stderr, "ERROR: Failed to open RTSP stream\n");
        av_dict_free(&opts);
        ret = -1;
        return -1;
    }
    av_dict_free(&opts);
    printf("✓ Connected successfully\n");
    
    // Discover stream information
    printf("Analyzing stream information...\n");
    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to find stream information\n");
        ret = -1;
        return -1;
    }
    printf("✓ Stream analysis complete\n");
    
    // Find video stream and calculate optimized dimensions
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            width = formatContext->streams[i]->codecpar->width;
            height = formatContext->streams[i]->codecpar->height;
            
            // Calculate hardware-optimized dimensions
            vir_width = RK_ALIGN_16(width);    // 16-byte alignment required
            vir_height = height;               // Height typically doesn't need alignment
            uv_height = (height + 1) / 2;     // UV plane height (handle odd heights)
            
            printf("✓ Video stream found:\n");
            printf("  - Resolution: %dx%d\n", width, height);
            printf("  - Aligned stride: %d\n", vir_width);
            printf("  - Codec: %s\n", avcodec_get_name(formatContext->streams[i]->codecpar->codec_id));
            printf("  - Input format: %s\n", av_get_pix_fmt_name((AVPixelFormat)formatContext->streams[i]->codecpar->format));

            break;
        }
    }
    
    if (videoStreamIndex == -1) {
        fprintf(stderr, "ERROR: No video stream found in input\n");
        ret = -1;
        return -1;
    }
    
    // Initialize video decoder
    codecContext = avcodec_alloc_context3(NULL);
    if (!codecContext) {
        fprintf(stderr, "ERROR: Failed to allocate codec context\n");
        ret = -1;
        return -1;
    }
    
    avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar);
    
    codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec) {
        fprintf(stderr, "ERROR: Decoder not found for codec\n");
        ret = -1;
        return -1;
    }
    
    if (avcodec_open2(codecContext, codec, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to open decoder\n");
        ret = -1;
        return -1;
    }
    printf("✓ Video decoder initialized\n");
    
    // Create swscale context for format conversion to NV12
    sws_ctx = sws_getContext(
        codecContext->width, codecContext->height, codecContext->pix_fmt,
        width, height, AV_PIX_FMT_NV12,  // Hardware-native NV12 format
        SWS_FAST_BILINEAR, NULL, NULL, NULL);
    
    if (!sws_ctx) {
        fprintf(stderr, "ERROR: Failed to create swscale context\n");
        ret = -1;
        return -1;
    }
    printf("✓ Format converter initialized (NV12 output)\n");
    
    // Allocate FFmpeg frames
    avframe = av_frame_alloc();
    if (!avframe) {
        fprintf(stderr, "ERROR: Failed to allocate input frame\n");
        ret = -1;
        return -1;
    }
    
    nv12_frame = av_frame_alloc();
    if (!nv12_frame) {
        fprintf(stderr, "ERROR: Failed to allocate NV12 frame\n");
        ret = -1;
        return -1;
    }
    
    // Configure NV12 frame properties
    nv12_frame->format = AV_PIX_FMT_NV12;
    nv12_frame->width = width;
    nv12_frame->height = height;
    if (av_frame_get_buffer(nv12_frame, 0) < 0) {
        fprintf(stderr, "ERROR: Failed to allocate NV12 frame buffer\n");
        ret = -1;
        return -1;
    }
    
    av_init_packet(&packet);
    
    // Initialize Rockchip MPI system
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        fprintf(stderr, "ERROR: Rockchip MPI system initialization failed\n");
        ret = -1;
        return -1;
    }
    printf("✓ Rockchip MPI system initialized\n");
    
    // Create optimized memory pool for NV12 data
    MB_POOL_CONFIG_S PoolCfg;
    memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
    PoolCfg.u64MBSize = vir_width * vir_height * 3 / 2;  // NV12: 1.5 bytes per pixel
    PoolCfg.u32MBCnt = 3;                                // Triple buffering for smooth pipeline
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;             // DMA-coherent memory
    
    src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    if (src_Pool == MB_INVALID_POOLID) {
        fprintf(stderr, "ERROR: Failed to create memory pool\n");
        ret = -1;
        return -1;
    }
    printf("✓ Memory pool created (size: %lu bytes, buffers: %d)\n", 
           PoolCfg.u64MBSize, PoolCfg.u32MBCnt);
    
    // Allocate memory block from pool
    src_Blk = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_TRUE);
    if (src_Blk == MB_INVALID_HANDLE) {
        fprintf(stderr, "ERROR: Failed to allocate memory block\n");
        ret = -1;
        return -1;
    }
    
    data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    if (!data) {
        fprintf(stderr, "ERROR: Failed to get virtual address for memory block\n");
        ret = -1;
        return -1;
    }
    printf("✓ Memory block allocated and mapped\n");
    
    // Configure video frame structure for VENC
    VIDEO_FRAME_INFO_S h264_frame;
    memset(&h264_frame, 0, sizeof(h264_frame));
    h264_frame.stVFrame.u32Width = width;                    // Actual image width
    h264_frame.stVFrame.u32Height = height;                  // Actual image height
    h264_frame.stVFrame.u32VirWidth = vir_width;             // Hardware-aligned stride
    h264_frame.stVFrame.u32VirHeight = vir_height;           // Virtual height
    h264_frame.stVFrame.enPixelFormat = RK_FMT_YUV420SP;     // NV12 format
    h264_frame.stVFrame.u32FrameFlag = 160;                  // Standard frame flag
    h264_frame.stVFrame.pMbBlk = src_Blk;                    // Memory block reference
    
    // Initialize RTSP output server
    g_rtsplive = create_rtsp_demo(554);
    if (!g_rtsplive) {
        fprintf(stderr, "ERROR: Failed to create RTSP server\n");
        ret = -1;
        return -1;
    }
    
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsp_session) {
        fprintf(stderr, "ERROR: Failed to create RTSP session\n");
        ret = -1;
        return -1;
    }
    
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("✓ RTSP server initialized on port 554\n");
    printf("  - Stream URL: rtsp://<device_ip>:554/live/0\n");
    
    // Initialize and start video encoder
    RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
    s32Ret = venc_init(0, width, height, enCodecType);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: Video encoder initialization failed: %#x\n", s32Ret);
        ret = -1;
        return -1;
    }
    printf("✓ Video encoder initialized\n");
    
    // Critical: Start encoder channel to begin frame processing
    VENC_RECV_PIC_PARAM_S stRecvParam;
    memset(&stRecvParam, 0, sizeof(stRecvParam));
    stRecvParam.s32RecvPicNum = -1;  // Process unlimited frames
    
    s32Ret = RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to start encoder channel: %#x\n", s32Ret);
        ret = -1;
        return -1;
    }
    printf("✓ Video encoder channel started\n");
    
    printf("\n========================================\n");
    printf("Starting frame processing pipeline...\n");
    printf("Press Ctrl+C to stop gracefully\n");
    printf("========================================\n\n");
    
    start_time = get_time_us();
    last_fps_time = start_time;
    
    // Main processing loop with error resilience
    while (g_running && av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            // Send packet to decoder
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                // Process all available decoded frames
                while (avcodec_receive_frame(codecContext, avframe) == 0) {
					// Safety check
                    if (!data) {
                        printf("ERROR: VENC input buffer is NULL!\n");
                        continue;
                    }
                    // Convert decoded frame to hardware-native NV12 format
                    sws_scale(sws_ctx,
                              (const uint8_t * const*)avframe->data, 
                              avframe->linesize,
                              0, 
                              codecContext->height,
                              nv12_frame->data, 
                              nv12_frame->linesize);
                    
                    // Copy Y plane (luminance) with stride alignment
                    for (int i = 0; i < height; i++) {
                        memcpy(data + i * vir_width,
                               nv12_frame->data[0] + i * nv12_frame->linesize[0],
                               width);
                    }
                    
                    // Copy UV plane (chrominance) - interleaved U and V samples
                    int uv_offset = vir_width * vir_height;
                    for (int i = 0; i < uv_height; i++) {
                        memcpy(data + uv_offset + i * vir_width,
                               nv12_frame->data[1] + i * nv12_frame->linesize[1],
                               width);  // UV width same as Y for NV12
                    }
                    
                    // Critical: Ensure hardware can see CPU-written data
                    RK_MPI_SYS_MmzFlushCache(src_Blk, RK_FALSE);
                    
                    // Update frame timing information
                    h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
                    h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
                    
                    // Send frame to hardware encoder
                    s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
                    if (s32Ret != RK_SUCCESS) {
                        printf("WARNING: VENC_SendFrame failed: %#x\n", s32Ret);
                        consecutive_errors++;
                        if (consecutive_errors > 10) {
                            fprintf(stderr, "ERROR: Too many consecutive encoder errors, exiting\n");
                            g_running = false;
                            break;
                        }
                        continue;
                    }
                    consecutive_errors = 0;  // Reset error counter on success
                    
                    // Retrieve encoded H.264 stream
                    // memset(&stFrame, 0, sizeof(stFrame));
                    s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, 1000);
                    
                    if (s32Ret == RK_SUCCESS) {
                        // Transmit encoded frame via RTSP
                        if (g_rtsplive && g_rtsp_session) {
                            void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                            rtsp_tx_video(g_rtsp_session, 
                                          (uint8_t *)pData, 
                                          stFrame.pstPack->u32Len,
                                          stFrame.pstPack->u64PTS);
                            rtsp_do_event(g_rtsplive);
                        }
                        
                        // Release encoded stream buffer
                        RK_MPI_VENC_ReleaseStream(0, &stFrame);
                    } else if (s32Ret != RK_ERR_VENC_BUF_EMPTY) {
                        printf("WARNING: VENC_GetStream failed: %#x\n", s32Ret);
                    }
                    
                    // Update performance counters
                    frame_count++;
                    fps_frame_count++;
                    
                    // Calculate and display performance metrics
                    uint64_t current_time = get_time_us();
                    uint64_t elapsed_us = current_time - last_fps_time;
                    
                    if (elapsed_us >= 1000000) {  // Every second
                        double fps = (double)fps_frame_count / (elapsed_us / 1000000.0);
                        double total_elapsed = (current_time - start_time) / 1000000.0;
                        
                        printf("[%.1fs] Frames: %lu | FPS: %.2f | Resolution: %dx%d | Stride: %d\n",
                               total_elapsed,
                               (unsigned long)frame_count,
                               fps,
                               width,
                               height,
                               vir_width);
                        
                        fps_frame_count = 0;
                        last_fps_time = current_time;
                    }
                }
            }
        }
        
        av_packet_unref(&packet);
    }
    
    // Display final performance statistics
    uint64_t end_time = get_time_us();
    double total_time = (end_time - start_time) / 1000000.0;
    double avg_fps = total_time > 0 ? (double)frame_count / total_time : 0;
    
    printf("\n========================================\n");
    printf("Processing Complete\n");
    printf("========================================\n");
    printf("Final Statistics:\n");
    printf("  - Total frames processed: %lu\n", (unsigned long)frame_count);
    printf("  - Total processing time: %.2f seconds\n", total_time);
    printf("  - Average FPS: %.2f\n", avg_fps);
    printf("  - Resolution: %dx%d (stride: %d)\n", width, height, vir_width);
    printf("========================================\n");


    printf("\nInitiating resource cleanup...\n");
	if (stFrame.pstPack) {
        free(stFrame.pstPack);
        printf("✓ VENC pack structure freed\n");
    }
    
    // Cleanup FFmpeg resources
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
    
    // Cleanup Rockchip MPI resources
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
    
    printf("\nCleanup complete. Exiting.\n");
 
    return ret;

}