#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#define RK_ALIGN_16(x) (((x) + 15) & (~15))
#define MODEL_WIDTH  640
#define MODEL_HEIGHT 640
#define RTSP_INPUT_URL "rtsp://220.254.72.200/Src/MediaInput/h264/stream_1"

static volatile bool g_running = true;
static float g_scale = 1.0f;
static int g_leftPadding = 0;
static int g_topPadding = 0;

// Global contexts
rknn_app_context_t rknn_app_ctx;
object_detect_result_list od_results;
rtsp_demo_handle g_rtsplive = NULL;
rtsp_session_handle g_rtsp_session = NULL;


/**
 * Render text labels using OpenCV on small regions
 * Much faster than full-frame BGR conversion
 */
void draw_text_labels_hybrid(unsigned char *yuv_data, int width, int height, int vir_width,
                            int x, int y, const char *text) {
    if (!text || strlen(text) == 0) return;
    
    // Calculate text dimensions
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    
    int padding = 6;
    int label_w = text_size.width + padding * 2;
    int label_h = text_size.height + padding * 2;
    
    // Position text above bounding box (or below if too high)
    int text_x = x;
    int text_y = (y > label_h + 5) ? (y - label_h - 2) : (y + 25);
    
    // Clamp to frame boundaries
    if (text_x + label_w > width) text_x = width - label_w;
    if (text_x < 0) text_x = 0;
    if (text_y + label_h > height) text_y = height - label_h;
    if (text_y < 0) text_y = 0;
    
    // Create small BGR image for text rendering
    cv::Mat label_img(label_h, label_w, CV_8UC3, cv::Scalar(0, 0, 0)); // Black background
    
    // Draw text with OpenCV (high quality)
    cv::putText(label_img, text, cv::Point(padding, label_h - padding), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    
    // Convert small BGR region to YUV and copy to main buffer
    unsigned char *y_plane = yuv_data;
    unsigned char *uv_plane = yuv_data + vir_width * height;
    
    for (int row = 0; row < label_h; row++) {
        if (text_y + row >= height) break;
        
        uint8_t *bgr_ptr = label_img.ptr<uint8_t>(row);
        int y_offset = (text_y + row) * vir_width + text_x;
        
        for (int col = 0; col < label_w; col++) {
            if (text_x + col >= width) break;
            
            // Convert BGR to Y (luminance) using standard formula
            uint8_t b = bgr_ptr[col * 3];
            uint8_t g = bgr_ptr[col * 3 + 1];
            uint8_t r = bgr_ptr[col * 3 + 2];
            
            // ITU-R BT.601 conversion: Y = 0.299*R + 0.587*G + 0.114*B
            uint8_t y_val = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
            
            y_plane[y_offset + col] = y_val;
        }
    }
    
    // Set UV components to neutral (128) for the text area to avoid color artifacts
    int uv_start_y = text_y / 2;
    int uv_end_y = (text_y + label_h) / 2;
    int uv_start_x = (text_x / 2) * 2; // Align to even pixel
    int uv_width = ((label_w + 1) / 2) * 2;
    
    for (int uv_row = uv_start_y; uv_row < uv_end_y && uv_row < height / 2; uv_row++) {
        int uv_offset = uv_row * vir_width + uv_start_x;
        for (int i = 0; i < uv_width && uv_start_x + i < vir_width; i += 2) {
            uv_plane[uv_offset + i] = 128;     // U (neutral)
            uv_plane[uv_offset + i + 1] = 128; // V (neutral)
        }
    }
}

void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down gracefully...\n", sig);
    g_running = false;
}

uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

void letterbox(cv::Mat& input, cv::Mat& output, int src_width, int src_height) {
    float scaleX = (float)MODEL_WIDTH / (float)src_width;
    float scaleY = (float)MODEL_HEIGHT / (float)src_height;
    g_scale = std::min(scaleX, scaleY);
    
    int inputWidth = (int)((float)src_width * g_scale);
    int inputHeight = (int)((float)src_height * g_scale);

    g_leftPadding = (MODEL_WIDTH - inputWidth) / 2;
    g_topPadding = (MODEL_HEIGHT - inputHeight) / 2;

    cv::Mat inputScale;
    cv::resize(input, inputScale, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_NEAREST);
    
    output.setTo(cv::Scalar(0, 0, 0));
    cv::Rect roi(g_leftPadding, g_topPadding, inputWidth, inputHeight);
    inputScale.copyTo(output(roi));
}

void mapCoordinates(int *x, int *y) {
    int mx = *x - g_leftPadding;
    int my = *y - g_topPadding;
    *x = (int)((float)mx / g_scale);
    *y = (int)((float)my / g_scale);
}

/**
 * I420 (Planar) format bounding box drawing
 * Fixes cyan artifacts by handling separate U/V planes correctly
 */
void draw_bbox_i420_correct(unsigned char *data, int width, int height, int vir_width,
                           int x, int y, int w, int h, int thickness) {
    // Bounds validation
    if (!data || thickness <= 0) return;
    if (x < 0 || y < 0 || w <= 0 || h <= 0) return;
    if (x >= width || y >= height) return;
    
    // Clamp to frame boundaries
    if (x + w > width) w = width - x;
    if (y + h > height) h = height - y;
    if (w < 10 || h < 10) return;
    
    thickness = std::min(thickness, std::min(w/4, h/4));
    
    // Green color values for I420
    const uint8_t Y_GREEN = 150;  // Luminance
    const uint8_t U_GREEN = 44;   // Chroma U
    const uint8_t V_GREEN = 21;   // Chroma V
    
    // *** I420 PLANAR LAYOUT POINTERS ***
    unsigned char *y_plane = data;                                    // Y plane
    unsigned char *u_plane = data + vir_width * height;               // U plane (separate)
    unsigned char *v_plane = data + vir_width * height + (vir_width * height / 4); // V plane (separate)
    
    // Draw Y plane borders (full resolution)
    for (int t = 0; t < thickness; t++) {
        // Top and bottom horizontal lines
        if (y + t < height) {
            memset(y_plane + (y + t) * vir_width + x, Y_GREEN, w);
        }
        if (y + h - 1 - t >= 0 && y + h - 1 - t < height) {
            memset(y_plane + (y + h - 1 - t) * vir_width + x, Y_GREEN, w);
        }
    }
    
    // Left and right vertical lines
    for (int row = y; row < y + h && row < height; row++) {
        for (int t = 0; t < thickness; t++) {
            if (x + t < width) {
                y_plane[row * vir_width + x + t] = Y_GREEN;
            }
            if (x + w - 1 - t >= 0 && x + w - 1 - t < width) {
                y_plane[row * vir_width + x + w - 1 - t] = Y_GREEN;
            }
        }
    }
    
    // *** DRAW U AND V PLANES SEPARATELY (I420 FORMAT) ***
    int uv_width = vir_width / 2;
    int uv_height = height / 2;
    int uv_x = x / 2;
    int uv_y = y / 2;
    int uv_w = w / 2;
    int uv_h = h / 2;
    int uv_thickness = std::max(1, thickness / 2);
    
    // Clamp UV coordinates
    if (uv_x + uv_w > uv_width) uv_w = uv_width - uv_x;
    if (uv_y + uv_h > uv_height) uv_h = uv_height - uv_y;
    
    // Draw U plane borders
    for (int t = 0; t < uv_thickness; t++) {
        if (uv_y + t < uv_height) {
            memset(u_plane + (uv_y + t) * uv_width + uv_x, U_GREEN, uv_w);
        }
        if (uv_y + uv_h - 1 - t >= 0 && uv_y + uv_h - 1 - t < uv_height) {
            memset(u_plane + (uv_y + uv_h - 1 - t) * uv_width + uv_x, U_GREEN, uv_w);
        }
    }
    
    for (int row = uv_y; row < uv_y + uv_h && row < uv_height; row++) {
        for (int t = 0; t < uv_thickness; t++) {
            if (uv_x + t < uv_width) {
                u_plane[row * uv_width + uv_x + t] = U_GREEN;
            }
            if (uv_x + uv_w - 1 - t >= 0 && uv_x + uv_w - 1 - t < uv_width) {
                u_plane[row * uv_width + uv_x + uv_w - 1 - t] = U_GREEN;
            }
        }
    }
    
    // Draw V plane borders (identical structure to U plane)
    for (int t = 0; t < uv_thickness; t++) {
        if (uv_y + t < uv_height) {
            memset(v_plane + (uv_y + t) * uv_width + uv_x, V_GREEN, uv_w);
        }
        if (uv_y + uv_h - 1 - t >= 0 && uv_y + uv_h - 1 - t < uv_height) {
            memset(v_plane + (uv_y + uv_h - 1 - t) * uv_width + uv_x, V_GREEN, uv_w);
        }
    }
    
    for (int row = uv_y; row < uv_y + uv_h && row < uv_height; row++) {
        for (int t = 0; t < uv_thickness; t++) {
            if (uv_x + t < uv_width) {
                v_plane[row * uv_width + uv_x + t] = V_GREEN;
            }
            if (uv_x + uv_w - 1 - t >= 0 && uv_x + uv_w - 1 - t < uv_width) {
                v_plane[row * uv_width + uv_x + uv_w - 1 - t] = V_GREEN;
            }
        }
    }
}

// VENC streaming thread
static void *GetMediaBuffer(void *arg) {
    (void)arg;
    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    stFrame.u32PackCount = 1;

    while (g_running) {
        RK_S32 s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, 1000);
        if (s32Ret == RK_SUCCESS) {
            if (g_rtsplive && g_rtsp_session) {
                void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, 
                              stFrame.pstPack->u32Len,
                              stFrame.pstPack->u64PTS);
                rtsp_do_event(g_rtsplive);
            }
            RK_MPI_VENC_ReleaseStream(0, &stFrame);
        }
        usleep(10000);
    }
    
    if (stFrame.pstPack) free(stFrame.pstPack);
    return NULL;
}

int main(int argc, char *argv[]) {
    system("RkLunch-stop.sh");
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    RK_S32 s32Ret = 0;
    cv::Mat letterbox_output(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
    
    const char *model_path = "./model/yolov5.rknn";
    unsigned char *temp_i420_buffer = NULL;
    cv::Mat *bgr_frame = NULL;
    
    VIDEO_FRAME_INFO_S h264_frame;
    VENC_RECV_PIC_PARAM_S stRecvParam;
    RK_U32 H264_TimeRef = 0;
    
    // FFmpeg variables
    AVFormatContext *formatContext = NULL;
    AVCodecContext *codecContext = NULL;
    AVCodec *codec = NULL;
    AVPacket packet;
    AVFrame *avframe = NULL;
    AVFrame *i420_frame = NULL;
    struct SwsContext *sws_ctx = NULL;
    int videoStreamIndex = -1;
    AVDictionary *opts = NULL;
    
    int width = 0, height = 0, vir_width = 0, vir_height = 0;
    
    // Memory pool variables
    MB_POOL src_Pool = MB_INVALID_POOLID;
    MB_BLK src_Blk = MB_INVALID_HANDLE;
    MB_POOL_CONFIG_S PoolCfg;
    unsigned char *data = NULL;
    
    uint64_t frame_count = 0, start_time = 0, last_fps_time = 0, fps_frame_count = 0;
    int consecutive_errors = 0;
    
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&h264_frame, 0, sizeof(h264_frame));
    memset(&PoolCfg, 0, sizeof(PoolCfg));
    memset(&stRecvParam, 0, sizeof(stRecvParam));
    
    printf("========================================\n");
    printf("RTSP YOLOv5 Optimized Software OSD\n");
    printf("Hardware OSD: Not available (RV1106 limitation)\n");
    printf("Target: 12-15 FPS (30-60%% improvement)\n");
    printf("========================================\n");
    
    // Initialize YOLOv5
    if (init_yolov5_model(model_path, &rknn_app_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize YOLOv5\n");
        return -1;
    }
    if (init_post_process() != 0) {
        fprintf(stderr, "ERROR: Failed to initialize post-processing\n");
        return -1;
    }
    printf("✓ YOLOv5 model initialized\n");
    
    // RTSP input setup
    avformat_network_init();
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    
    printf("Connecting to: %s\n", RTSP_INPUT_URL);
    if (avformat_open_input(&formatContext, RTSP_INPUT_URL, NULL, &opts) != 0) {
        fprintf(stderr, "ERROR: Failed to open RTSP stream\n");
        av_dict_free(&opts);
        return -1;
    }
    av_dict_free(&opts);
    printf("✓ Connected successfully\n");
    
    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to find stream information\n");
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
            printf("✓ Video stream: %dx%d (stride: %d)\n", width, height, vir_width);
            break;
        }
    }
    
    if (videoStreamIndex == -1) {
        fprintf(stderr, "ERROR: No video stream found\n");
        return -1;
    }
    
    // Decoder setup
    codecContext = avcodec_alloc_context3(NULL);
    if (!codecContext) {
        fprintf(stderr, "ERROR: Failed to allocate codec context\n");
        return -1;
    }
    
    avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar);
    codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec || avcodec_open2(codecContext, codec, NULL) < 0) {
        fprintf(stderr, "ERROR: Decoder initialization failed\n");
        return -1;
    }
    printf("✓ Video decoder initialized\n");
    
    sws_ctx = sws_getContext(
        codecContext->width, codecContext->height, codecContext->pix_fmt,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) {
        fprintf(stderr, "ERROR: Failed to create SWS context\n");
        return -1;
    }
    printf("✓ Format converter initialized\n");
    
    avframe = av_frame_alloc();
    i420_frame = av_frame_alloc();
    if (!avframe || !i420_frame) {
        fprintf(stderr, "ERROR: Failed to allocate frames\n");
        return -1;
    }
    
    i420_frame->format = AV_PIX_FMT_YUV420P;
    i420_frame->width = width;
    i420_frame->height = height;
    if (av_frame_get_buffer(i420_frame, 32) < 0) {
        fprintf(stderr, "ERROR: Failed to allocate I420 frame buffer\n");
        return -1;
    }
    
    av_init_packet(&packet);
    
    // Initialize MPI system (no VI needed for software OSD)
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        fprintf(stderr, "ERROR: RK_MPI_SYS_Init failed\n");
        return -1;
    }
    printf("✓ MPI system initialized\n");
    
    // Initialize VENC
    s32Ret = venc_init(0, width, height, RK_VIDEO_ID_AVC);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: VENC init failed: 0x%x\n", s32Ret);
        return -1;
    }
    printf("✓ VENC initialized\n");
    
    // Create memory pool
    PoolCfg.u64MBSize = vir_width * vir_height * 3 / 2;
    PoolCfg.u32MBCnt = 3;
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
    
    src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    if (src_Pool == MB_INVALID_POOLID) {
        fprintf(stderr, "ERROR: Failed to create memory pool\n");
        return -1;
    }
    
    src_Blk = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_TRUE);
    if (src_Blk == MB_INVALID_HANDLE) {
        fprintf(stderr, "ERROR: Failed to get memory block\n");
        return -1;
    }
    
    data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    if (!data) {
        fprintf(stderr, "ERROR: Failed to get virtual address\n");
        return -1;
    }
    printf("✓ Memory pool created (%lu bytes)\n", PoolCfg.u64MBSize);
    
    // Configure frame structure
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = vir_width;
    h264_frame.stVFrame.u32VirHeight = vir_height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_YUV420P;
    h264_frame.stVFrame.u32FrameFlag = 160;
    h264_frame.stVFrame.pMbBlk = src_Blk;
    
    // RTSP server setup
    g_rtsplive = create_rtsp_demo(554);
    if (!g_rtsplive) {
        fprintf(stderr, "ERROR: Failed to create RTSP demo\n");
        return -1;
    }
    
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsp_session) {
        fprintf(stderr, "ERROR: Failed to create RTSP session\n");
        return -1;
    }
    
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("✓ RTSP server: rtsp://<device_ip>:554/live/0\n");
    
    stRecvParam.s32RecvPicNum = -1;
    s32Ret = RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to start VENC receiving: 0x%x\n", s32Ret);
        return -1;
    }
    printf("✓ Video encoder started\n");
    
    // Start streaming thread
    pthread_t stream_thread;
    if (pthread_create(&stream_thread, NULL, GetMediaBuffer, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to create streaming thread\n");
        return -1;
    }
    
    // Allocate processing buffers
    temp_i420_buffer = (unsigned char *)malloc(width * height * 3 / 2);
    if (!temp_i420_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate temporary buffer\n");
        return -1;
    }
    
    bgr_frame = new cv::Mat(height, width, CV_8UC3);
    
    printf("\n========================================\n");
    printf("Starting Optimized Software OSD Pipeline\n");
    printf("Baseline: 9.5 FPS → Target: 12-15 FPS\n");
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");
    
    start_time = get_time_us();
    last_fps_time = start_time;
    
    // Main processing loop
    while (g_running && av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                while (avcodec_receive_frame(codecContext, avframe) == 0) {
                    if (!data) continue;
                    
                    uint64_t frame_start = get_time_us();
                    
                    // 1. Decode RTSP to I420
                    sws_scale(sws_ctx,
                              (const uint8_t * const*)avframe->data, 
                              avframe->linesize,
                              0, codecContext->height,
                              i420_frame->data, i420_frame->linesize);
                    
                    // 2. Copy I420 to VENC buffer with stride alignment (display path)
                    for (int i = 0; i < height; i++) {
                        memcpy(data + i * vir_width, 
                               i420_frame->data[0] + i * i420_frame->linesize[0], 
                               width);
                    }
                    
                    int venc_u_offset = vir_width * vir_height;
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(data + venc_u_offset + i * (vir_width / 2), 
                               i420_frame->data[1] + i * i420_frame->linesize[1], 
                               width / 2);
                    }
                    
                    int venc_v_offset = venc_u_offset + (vir_width * vir_height / 4);
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(data + venc_v_offset + i * (vir_width / 2), 
                               i420_frame->data[2] + i * i420_frame->linesize[2], 
                               width / 2);
                    }
                    
                    // 3. Prepare AI processing buffer (remove stride for OpenCV)
                    for (int i = 0; i < height; i++) {
                        memcpy(temp_i420_buffer + i * width,
                               data + i * vir_width, width);
                    }
                    
                    int temp_u_offset = width * height;
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(temp_i420_buffer + temp_u_offset + i * (width / 2),
                               data + venc_u_offset + i * (vir_width / 2), width / 2);
                    }
                    
                    int temp_v_offset = temp_u_offset + (width * height / 4);
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(temp_i420_buffer + temp_v_offset + i * (width / 2),
                               data + venc_v_offset + i * (vir_width / 2), width / 2);
                    }
                    
                    // 4. AI processing (convert to BGR only for inference)
                    static int ai_frame_counter = 0;
                    static object_detect_result_list cached_detections;
                    static bool cache_initialized = false;
                    
                    ai_frame_counter++;
                    if (ai_frame_counter % 3 == 0) {
                        cv::Mat i420_mat(height + height/2, width, CV_8UC1, temp_i420_buffer);
                        cv::cvtColor(i420_mat, *bgr_frame, cv::COLOR_YUV2BGR_I420);
                    
                        letterbox(*bgr_frame, letterbox_output, width, height);
                    
                        // 5. RKNN inference
                        memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letterbox_output.data, 
                           MODEL_WIDTH * MODEL_HEIGHT * 3);
                        memset(&od_results, 0, sizeof(object_detect_result_list));
      
                        inference_yolov5_model(&rknn_app_ctx, &od_results);

                        // Cache results for next frame
                        memcpy(&cached_detections, &od_results, sizeof(object_detect_result_list));
                        cache_initialized = true;
                    } else if(cache_initialized) {
                        // Skip AI processing, reuse cached results
                        memcpy(&od_results, &cached_detections, sizeof(object_detect_result_list));
                    }
                    uint64_t ai_time = get_time_us() - frame_start;
                    
                    // 6. *** OPTIMIZED SOFTWARE OSD (Direct YUV Drawing) ***
                    uint64_t osd_start = get_time_us();

                    for (int i = 0; i < od_results.count; i++) {
                        object_detect_result *det = &(od_results.results[i]);
                        // 1. Confidence threshold (increase from ~0.25 to 0.45)
                        if (det->prop < 0.45f) continue;
                        // 2. Valid class ID
                        if (det->cls_id < 0 || det->cls_id >= 80) continue;

                        int sX = (int)(det->box.left);
                        int sY = (int)(det->box.top);
                        int eX = (int)(det->box.right);
                        int eY = (int)(det->box.bottom);

                        if (sX >= eX || sY >= eY) continue;
                        if (sX < 0 || sY < 0 || eX >= MODEL_WIDTH || eY >= MODEL_HEIGHT) continue;
                        
                        // Map coordinates back to display space
                        mapCoordinates(&sX, &sY);
                        mapCoordinates(&eX, &eY);
                        
                        // Clamp to frame bounds
                        sX = std::max(0, std::min(sX, width - 1));
                        sY = std::max(0, std::min(sY, height - 1));
                        eX = std::max(0, std::min(eX, width - 1));
                        eY = std::max(0, std::min(eY, height - 1));
                        
                        int bbox_w = eX - sX;
                        int bbox_h = eY - sY;
                        
                        if (bbox_w > 10 && bbox_h > 10) {
                            // Draw bounding box (fast YUV method)
                            draw_bbox_i420_correct(data, width, height, vir_width,
                                                sX, sY, bbox_w, bbox_h, 3);
                            
                            // Draw text label (hybrid OpenCV method)
                            char label[64];
                            snprintf(label, sizeof(label), "%s %.0f%%", 
                                    coco_cls_to_name(det->cls_id), 
                                    det->prop * 100);
                            
                            draw_text_labels_hybrid(data, width, height, vir_width, sX, sY, label);
                        }
                    }

                    uint64_t osd_time = get_time_us() - osd_start;
                    
                    // 7. Send to VENC
                    RK_MPI_SYS_MmzFlushCache(src_Blk, RK_FALSE);
                    h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
                    h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
                    
                    s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
                    if (s32Ret != RK_SUCCESS) {
                        consecutive_errors++;
                        if (consecutive_errors > 10) {
                            printf("ERROR: Too many consecutive VENC errors, stopping\n");
                            g_running = false;
                            break;
                        }
                        printf("WARN: VENC_SendFrame failed: 0x%x\n", s32Ret);
                        continue;
                    }
                    consecutive_errors = 0;
                    
                    // 8. Performance monitoring
                    frame_count++;
                    fps_frame_count++;
                    
                    uint64_t current_time = get_time_us();
                    if (current_time - last_fps_time >= 1000000) {
                        double fps = (double)fps_frame_count / 
                                    ((current_time - last_fps_time) / 1000000.0);
                        double total_elapsed = (current_time - start_time) / 1000000.0;
                        double improvement = fps / 9.5;
                        double ai_ms = ai_time / 1000.0;
                        double osd_ms = osd_time / 1000.0;
                        
                        const char* status = fps >= 12.0 ? "✓ TARGET" : "";
                        
                        printf("[%.1fs] FPS: %.2f | AI: %.1fms | OSD: %.1fms | Det: %d | Gain: %.1fx %s\n",
                               total_elapsed, fps, ai_ms, osd_ms, od_results.count, improvement, status);
                        
                        fps_frame_count = 0;
                        last_fps_time = current_time;
                    }
                }
            }
        }
        av_packet_unref(&packet);
    }
    
    // Performance summary
    uint64_t end_time = get_time_us();
    double total_time = (end_time - start_time) / 1000000.0;
    double avg_fps = total_time > 0 ? (double)frame_count / total_time : 0;
    
    printf("\n========================================\n");
    printf("Final Performance Results\n");
    printf("========================================\n");
    printf("  Total frames: %lu\n", (unsigned long)frame_count);
    printf("  Total time: %.2f seconds\n", total_time);
    printf("  Average FPS: %.2f\n", avg_fps);
    printf("  Baseline: 9.5 FPS\n");
    printf("  Improvement: %.1fx faster (%.0f%% gain)\n", 
           avg_fps / 9.5, (avg_fps / 9.5 - 1.0) * 100);
    printf("  Target (12 FPS): %s\n", avg_fps >= 12.0 ? "ACHIEVED ✓" : "Close");
    printf("\n  Note: Hardware OSD unavailable for RTSP input\n");
    printf("        due to RV1106 silicon architecture.\n");
    printf("        This is the optimal software solution.\n");
    printf("========================================\n");
    
    // Comprehensive cleanup
    printf("\nCleaning up...\n");
    
    g_running = false;
    pthread_join(stream_thread, NULL);
    
    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);
    
    if (bgr_frame) delete bgr_frame;
    if (temp_i420_buffer) free(temp_i420_buffer);
    
    if (i420_frame) av_frame_free(&i420_frame);
    if (avframe) av_frame_free(&avframe);
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (codecContext) avcodec_free_context(&codecContext);
    if (formatContext) avformat_close_input(&formatContext);
    avformat_network_deinit();
    
    if (src_Blk != MB_INVALID_HANDLE) RK_MPI_MB_ReleaseMB(src_Blk);
    if (src_Pool != MB_INVALID_POOLID) RK_MPI_MB_DestroyPool(src_Pool);
    
    if (g_rtsplive) rtsp_del_demo(g_rtsplive);
    
    RK_MPI_SYS_Exit();
    
    release_yolov5_model(&rknn_app_ctx);
    deinit_post_process();
    
    printf("✓ Cleanup complete\n");
    
    return 0;
}