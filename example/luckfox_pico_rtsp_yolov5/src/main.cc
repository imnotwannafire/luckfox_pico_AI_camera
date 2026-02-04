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
#include <algorithm>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#define RK_ALIGN_16(x) (((x) + 15) & (~15))
#define MODEL_WIDTH  640
#define MODEL_HEIGHT 640
#define RTSP_INPUT_URL "rtsp://220.254.72.200/Src/MediaInput/h264/stream_1"

static volatile bool g_running = true;
static float g_scale = 1.0f;
static int g_leftPadding = 0;
static int g_topPadding = 0;

void signal_handler(int sig) {
    printf("\nReceived signal %d, initiating graceful shutdown...\n", sig);
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

int main(int argc, char *argv[]) {
    system("RkLunch-stop.sh");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    RK_S32 s32Ret = 0;
    int ret = 0;
    cv::Mat letterbox_output(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
    
    rknn_app_context_t rknn_app_ctx;
    object_detect_result_list od_results;
    const char *model_path = "./model/yolov5.rknn";
    char text_buffer[256];
    
    unsigned char *temp_i420_buffer = NULL;
    cv::Mat *bgr_frame = NULL;
    
    VENC_STREAM_S stFrame;
    VIDEO_FRAME_INFO_S h264_frame;
    VENC_RECV_PIC_PARAM_S stRecvParam;
    RK_U32 H264_TimeRef = 0;
    
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
    
    MB_POOL src_Pool = MB_INVALID_POOLID;
    MB_BLK src_Blk = MB_INVALID_HANDLE;
    MB_POOL_CONFIG_S PoolCfg;
    unsigned char *data = NULL;
    
    rtsp_demo_handle g_rtsplive = NULL;
    rtsp_session_handle g_rtsp_session = NULL;
    
    uint64_t frame_count = 0, start_time = 0, last_fps_time = 0, fps_frame_count = 0;
    int consecutive_errors = 0;
    
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&stFrame, 0, sizeof(stFrame));
    memset(&h264_frame, 0, sizeof(h264_frame));
    memset(&PoolCfg, 0, sizeof(PoolCfg));
    memset(&stRecvParam, 0, sizeof(stRecvParam));
    
    printf("========================================\n");
    printf("RTSP YOLOv5 Object Detection System\n");
    printf("========================================\n");
    
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    if (!stFrame.pstPack) {
        fprintf(stderr, "ERROR: Failed to allocate VENC_PACK_S\n");
        return -1;
    }
    stFrame.u32PackCount = 1;
    
    if (init_yolov5_model(model_path, &rknn_app_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize YOLOv5 model\n");
        return -1;
    }
    
    if (init_post_process() != 0) {
        fprintf(stderr, "ERROR: Failed to initialize post-processing\n");
        return -1;
    }
    printf("✓ YOLOv5 model and post-processing initialized\n");
    
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
    
    cv::Mat i420_output;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            width = formatContext->streams[i]->codecpar->width;
            height = formatContext->streams[i]->codecpar->height;
            i420_output.create(height + height/2, width, CV_8UC1);
            
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
    
    codecContext = avcodec_alloc_context3(NULL);
    if (!codecContext) return -1;
    
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
    
    if (!sws_ctx) return -1;
    printf("✓ Format converter initialized (I420 output)\n");
    
    avframe = av_frame_alloc();
    i420_frame = av_frame_alloc();
    if (!avframe || !i420_frame) return -1;
    
    i420_frame->format = AV_PIX_FMT_YUV420P;
    i420_frame->width = width;
    i420_frame->height = height;
    if (av_frame_get_buffer(i420_frame, 32) < 0) {
        fprintf(stderr, "ERROR: Failed to allocate I420 frame buffer\n");
        return -1;
    }
    
    av_init_packet(&packet);
    
    if (RK_MPI_SYS_Init() != RK_SUCCESS) return -1;
    printf("✓ Rockchip MPI system initialized\n");
    
    PoolCfg.u64MBSize = vir_width * vir_height * 3 / 2;
    PoolCfg.u32MBCnt = 3;
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
    
    src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    if (src_Pool == MB_INVALID_POOLID) return -1;
    
    src_Blk = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_TRUE);
    if (src_Blk == MB_INVALID_HANDLE) return -1;
    
    data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    if (!data) return -1;
    printf("✓ Memory pool created (%lu bytes, %d buffers)\n", 
           PoolCfg.u64MBSize, PoolCfg.u32MBCnt);
    
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = vir_width;
    h264_frame.stVFrame.u32VirHeight = vir_height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_YUV420P;
    h264_frame.stVFrame.u32FrameFlag = 160;
    h264_frame.stVFrame.pMbBlk = src_Blk;
    
    g_rtsplive = create_rtsp_demo(554);
    if (!g_rtsplive) return -1;
    
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsp_session) return -1;
    
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("✓ RTSP server initialized on port 554\n");
    printf("  - Stream URL: rtsp://<device_ip>:554/live/0\n");
    
    s32Ret = venc_init(0, width, height, RK_VIDEO_ID_AVC);
    if (s32Ret != RK_SUCCESS) return -1;
    
    stRecvParam.s32RecvPicNum = -1;
    s32Ret = RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);
    if (s32Ret != RK_SUCCESS) return -1;
    printf("✓ Video encoder channel started\n");
    
    temp_i420_buffer = (unsigned char *)malloc(width * height * 3 / 2);
    if (!temp_i420_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate temporary I420 buffer\n");
        return -1;
    }
    
    bgr_frame = new cv::Mat(height, width, CV_8UC3);
    
    printf("\n========================================\n");
    printf("Starting YOLOv5 object detection pipeline...\n");
    printf("Press Ctrl+C to stop gracefully\n");
    printf("========================================\n\n");
    
    start_time = get_time_us();
    last_fps_time = start_time;
    
    while (g_running && av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                while (avcodec_receive_frame(codecContext, avframe) == 0) {
                    if (!data) continue;
                    
                    sws_scale(sws_ctx,
                              (const uint8_t * const*)avframe->data, 
                              avframe->linesize,
                              0, codecContext->height,
                              i420_frame->data, i420_frame->linesize);
                    
                    // Copy to VENC buffer with stride
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
                    
                    // *** AI PROCESSING ***
                    // Remove stride for OpenCV
                    for (int i = 0; i < height; i++) {
                        memcpy(temp_i420_buffer + i * width,
                               data + i * vir_width,
                               width);
                    }
                    
                    int temp_u_offset = width * height;
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(temp_i420_buffer + temp_u_offset + i * (width / 2),
                               data + venc_u_offset + i * (vir_width / 2),
                               width / 2);
                    }
                    
                    int temp_v_offset = temp_u_offset + (width * height / 4);
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(temp_i420_buffer + temp_v_offset + i * (width / 2),
                               data + venc_v_offset + i * (vir_width / 2),
                               width / 2);
                    }
                    
                    cv::Mat i420_mat(height + height/2, width, CV_8UC1, temp_i420_buffer);
                    cv::cvtColor(i420_mat, *bgr_frame, cv::COLOR_YUV2BGR_I420);
                    
                    letterbox(*bgr_frame, letterbox_output, width, height);
                    
                    memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letterbox_output.data, 
                           MODEL_WIDTH * MODEL_HEIGHT * 3);
                    inference_yolov5_model(&rknn_app_ctx, &od_results);
                    
                    for (int i = 0; i < od_results.count; i++) {
                        object_detect_result *det_result = &(od_results.results[i]);
                        
                        int sX = (int)(det_result->box.left);
                        int sY = (int)(det_result->box.top);
                        int eX = (int)(det_result->box.right);
                        int eY = (int)(det_result->box.bottom);
                        
                        mapCoordinates(&sX, &sY);
                        mapCoordinates(&eX, &eY);
                        
                        sX = std::max(0, std::min(sX, width-1));
                        sY = std::max(0, std::min(sY, height-1));
                        eX = std::max(0, std::min(eX, width-1));
                        eY = std::max(0, std::min(eY, height-1));
                        
                        cv::rectangle(*bgr_frame, cv::Point(sX, sY), cv::Point(eX, eY), 
                                      cv::Scalar(0, 255, 0), 3);
                        
                        snprintf(text_buffer, sizeof(text_buffer), "%s %.1f%%", 
                                coco_cls_to_name(det_result->cls_id), 
                                det_result->prop * 100);
                        cv::putText(*bgr_frame, text_buffer, cv::Point(sX, sY - 8),
                                    cv::FONT_HERSHEY_SIMPLEX, 1,
                                    cv::Scalar(0, 255, 0), 2);
                    }
                    
                    cv::cvtColor(*bgr_frame, i420_output, cv::COLOR_BGR2YUV_I420);
                    
                    // Copy back to VENC buffer with stride
                    for (int i = 0; i < height; i++) {
                        memcpy(data + i * vir_width, 
                               i420_output.data + i * width, 
                               width);
                    }
                    
                    int src_u_offset = width * height;
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(data + venc_u_offset + i * (vir_width / 2),
                               i420_output.data + src_u_offset + i * (width / 2),
                               width / 2);
                    }
                    
                    int src_v_offset = src_u_offset + (width * height / 4);
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(data + venc_v_offset + i * (vir_width / 2),
                               i420_output.data + src_v_offset + i * (width / 2),
                               width / 2);
                    }
                    
                    RK_MPI_SYS_MmzFlushCache(src_Blk, RK_FALSE);
                    
                    h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
                    h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
                    
                    s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
                    if (s32Ret != RK_SUCCESS) {
                        consecutive_errors++;
                        if (consecutive_errors > 10) {
                            g_running = false;
                            break;
                        }
                        continue;
                    }
                    consecutive_errors = 0;
                    
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
                    }
                    
                    frame_count++;
                    fps_frame_count++;
                    
                    uint64_t current_time = get_time_us();
                    uint64_t elapsed_us = current_time - last_fps_time;
                    
                    if (elapsed_us >= 1000000) {
                        double fps = (double)fps_frame_count / (elapsed_us / 1000000.0);
                        double total_elapsed = (current_time - start_time) / 1000000.0;
                        
                        printf("[%.1fs] Frames: %lu | FPS: %.2f | Detections: %d\n",
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
    
    uint64_t end_time = get_time_us();
    double total_time = (end_time - start_time) / 1000000.0;
    double avg_fps = total_time > 0 ? (double)frame_count / total_time : 0;
    
    printf("\n========================================\n");
    printf("YOLOv5 Processing Complete\n");
    printf("========================================\n");
    printf("  - Total frames: %lu\n", (unsigned long)frame_count);
    printf("  - Total time: %.2f seconds\n", total_time);
    printf("  - Average FPS: %.2f\n", avg_fps);
    printf("  - Resolution: %dx%d (stride: %d)\n", width, height, vir_width);
    printf("========================================\n");
    
    printf("\nCleaning up...\n");
    
    if (bgr_frame) delete bgr_frame;
    if (temp_i420_buffer) free(temp_i420_buffer);
    if (stFrame.pstPack) free(stFrame.pstPack);
    
    if (i420_frame) av_frame_free(&i420_frame);
    if (avframe) av_frame_free(&avframe);
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (codecContext) avcodec_free_context(&codecContext);
    if (formatContext) avformat_close_input(&formatContext);
    avformat_network_deinit();
    
    if (src_Blk != MB_INVALID_HANDLE) RK_MPI_MB_ReleaseMB(src_Blk);
    if (src_Pool != MB_INVALID_POOLID) RK_MPI_MB_DestroyPool(src_Pool);
    
    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);
    
    if (g_rtsplive) rtsp_del_demo(g_rtsplive);
    
    RK_MPI_SYS_Exit();
    
    release_yolov5_model(&rknn_app_ctx);
    deinit_post_process();
    
    printf("✓ Cleanup complete\n");
    
    return ret;
}