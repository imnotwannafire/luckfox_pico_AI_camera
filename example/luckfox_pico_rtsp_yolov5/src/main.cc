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
#include <map>
#include <set> 

#include "BYTETracker.h"
#include "dataType.h"
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

// Global tracker variables
static BYTETracker* g_tracker = nullptr;
static std::vector<STrack> g_active_tracks;
static std::map<int, int> g_track_labels;

void cleanup_stale_labels(const std::vector<STrack>& active_tracks) {
    std::set<int> active_ids;
    for (const auto& track : active_tracks) {
        active_ids.insert(track.track_id);
    }
    
    for (auto it = g_track_labels.begin(); it != g_track_labels.end();) {
        if (active_ids.find(it->first) == active_ids.end()) {
            it = g_track_labels.erase(it);
        } else {
            ++it;
        }
    }
}

void draw_text_labels_hybrid(unsigned char *yuv_data, int width, int height, int vir_width,
                            int x, int y, const char *text) {
    if (!text || strlen(text) == 0) return;
    
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
    
    int padding = 5;
    int label_w = text_size.width + padding * 2;
    int label_h = text_size.height + padding * 2;
    
    int text_x = x;
    int text_y = (y > label_h + 5) ? (y - label_h - 2) : (y + 20);
    
    // Clamp to frame boundaries
    text_x = std::max(0, std::min(text_x, width - label_w));
    text_y = std::max(0, std::min(text_y, height - label_h));
    
    cv::Mat label_img(label_h, label_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(label_img, text, cv::Point(padding, label_h - padding), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    
    unsigned char *y_plane = yuv_data;
    unsigned char *uv_plane = yuv_data + vir_width * height;
    
    // Convert BGR to Y (luminance)
    for (int row = 0; row < label_h && text_y + row < height; row++) {
        uint8_t *bgr_ptr = label_img.ptr<uint8_t>(row);
        int y_offset = (text_y + row) * vir_width + text_x;
        
        for (int col = 0; col < label_w && text_x + col < width; col++) {
            uint8_t b = bgr_ptr[col * 3];
            uint8_t g = bgr_ptr[col * 3 + 1];
            uint8_t r = bgr_ptr[col * 3 + 2];
            y_plane[y_offset + col] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
    
    // Set UV to neutral for clean appearance
    int uv_start_y = text_y / 2;
    int uv_end_y = std::min((text_y + label_h) / 2, height / 2);
    int uv_start_x = (text_x / 2) * 2;
    int uv_width = ((label_w + 1) / 2) * 2;
    
    for (int uv_row = uv_start_y; uv_row < uv_end_y; uv_row++) {
        int uv_offset = uv_row * vir_width + uv_start_x;
        for (int i = 0; i < uv_width && uv_start_x + i < vir_width; i += 2) {
            uv_plane[uv_offset + i] = 128;     // U neutral
            uv_plane[uv_offset + i + 1] = 128; // V neutral
        }
    }
}

void signal_handler(int sig) {
    printf("\nShutting down gracefully...\n");
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

void draw_bbox_i420_correct(unsigned char *data, int width, int height, int vir_width,
                           int x, int y, int w, int h, int thickness) {
    if (!data || thickness <= 0 || x < 0 || y < 0 || w <= 0 || h <= 0) return;
    if (x >= width || y >= height) return;
    
    w = std::min(w, width - x);
    h = std::min(h, height - y);
    if (w < 10 || h < 10) return;
    
    thickness = std::min(thickness, std::min(w/4, h/4));
    
    const uint8_t Y_GREEN = 150, U_GREEN = 44, V_GREEN = 21;
    
    unsigned char *y_plane = data;
    unsigned char *u_plane = data + vir_width * height;
    unsigned char *v_plane = data + vir_width * height + (vir_width * height / 4);
    
    // Draw Y plane borders
    for (int t = 0; t < thickness; t++) {
        if (y + t < height) {
            memset(y_plane + (y + t) * vir_width + x, Y_GREEN, w);
        }
        if (y + h - 1 - t >= 0 && y + h - 1 - t < height) {
            memset(y_plane + (y + h - 1 - t) * vir_width + x, Y_GREEN, w);
        }
    }
    
    for (int row = y; row < y + h && row < height; row++) {
        for (int t = 0; t < thickness; t++) {
            if (x + t < width) y_plane[row * vir_width + x + t] = Y_GREEN;
            if (x + w - 1 - t >= 0 && x + w - 1 - t < width) {
                y_plane[row * vir_width + x + w - 1 - t] = Y_GREEN;
            }
        }
    }
    
    // Draw UV planes
    int uv_width = vir_width / 2;
    int uv_height = height / 2;
    int uv_x = x / 2, uv_y = y / 2;
    int uv_w = std::min(w / 2, uv_width - uv_x);
    int uv_h = std::min(h / 2, uv_height - uv_y);
    int uv_thickness = std::max(1, thickness / 2);
    
    auto draw_uv_plane = [&](unsigned char* plane, uint8_t color) {
        for (int t = 0; t < uv_thickness; t++) {
            if (uv_y + t < uv_height) {
                memset(plane + (uv_y + t) * uv_width + uv_x, color, uv_w);
            }
            if (uv_y + uv_h - 1 - t >= 0 && uv_y + uv_h - 1 - t < uv_height) {
                memset(plane + (uv_y + uv_h - 1 - t) * uv_width + uv_x, color, uv_w);
            }
        }
        for (int row = uv_y; row < uv_y + uv_h && row < uv_height; row++) {
            for (int t = 0; t < uv_thickness; t++) {
                if (uv_x + t < uv_width) plane[row * uv_width + uv_x + t] = color;
                if (uv_x + uv_w - 1 - t >= 0 && uv_x + uv_w - 1 - t < uv_width) {
                    plane[row * uv_width + uv_x + uv_w - 1 - t] = color;
                }
            }
        }
    };
    
    draw_uv_plane(u_plane, U_GREEN);
    draw_uv_plane(v_plane, V_GREEN);
}

static void *GetMediaBuffer(void *arg) {
    (void)arg;
    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    if (!stFrame.pstPack) return NULL;
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
    
    free(stFrame.pstPack);
    return NULL;
}

std::vector<Object> convert_detections_to_tracker(
    const object_detect_result_list& od_results, 
    int width, int height) {
    
    std::vector<Object> objects;
    objects.reserve(od_results.count);
    
    for (int i = 0; i < od_results.count; i++) {
        const object_detect_result& det = od_results.results[i];
        
        if (det.prop < 0.20f) continue;
        if (det.cls_id < 0 || det.cls_id >= 80) continue;
        
        int sX = (int)(det.box.left);
        int sY = (int)(det.box.top);
        int eX = (int)(det.box.right);
        int eY = (int)(det.box.bottom);
        
        if (sX >= eX || sY >= eY) continue;
        if (sX < 0 || sY < 0 || eX >= MODEL_WIDTH || eY >= MODEL_HEIGHT) continue;
        
        mapCoordinates(&sX, &sY);
        mapCoordinates(&eX, &eY);
        
        sX = std::max(0, std::min(sX, width - 1));
        sY = std::max(0, std::min(sY, height - 1));
        eX = std::max(0, std::min(eX, width - 1));
        eY = std::max(0, std::min(eY, height - 1));
        
        int bbox_w = eX - sX;
        int bbox_h = eY - sY;
        
        if (bbox_w <= 8 || bbox_h <= 8) continue;
        
        Object obj;
        obj.rect = cv::Rect_<float>(
            static_cast<float>(sX), static_cast<float>(sY),
            static_cast<float>(bbox_w), static_cast<float>(bbox_h)
        );
        obj.label = det.cls_id;
        obj.prob = det.prop;
        objects.push_back(obj);
    }
    
    return objects;
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
    
    AVFormatContext *formatContext = NULL;
    AVCodecContext *codecContext = NULL;
    AVCodec *codec = NULL;
    AVPacket *packet = NULL;
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
    
    uint64_t frame_count = 0, start_time = 0, last_fps_time = 0, fps_frame_count = 0;
    int consecutive_errors = 0;
    
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&h264_frame, 0, sizeof(h264_frame));
    memset(&PoolCfg, 0, sizeof(PoolCfg));
    memset(&stRecvParam, 0, sizeof(stRecvParam));
    
    printf("========================================\n");
    printf("RTSP YOLOv5 + ByteTrack Object Tracking\n");
    printf("Target: 12-15 FPS with smooth display\n");
    printf("========================================\n");
    
    // Initialize YOLOv5
    if (init_yolov5_model(model_path, &rknn_app_ctx) != 0 || init_post_process() != 0) {
        fprintf(stderr, "ERROR: Failed to initialize YOLOv5\n");
        return -1;
    }
    printf("✓ YOLOv5 initialized\n");
    
    // Initialize ByteTracker (fps=5 matches effective detection rate: 15fps/3)
    g_tracker = new BYTETracker(5, 15);
    if (!g_tracker) {
        fprintf(stderr, "ERROR: Failed to initialize ByteTracker\n");
        return -1;
    }
    printf("✓ ByteTracker initialized\n");
    
    // RTSP input setup
    avformat_network_init();
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    
    if (avformat_open_input(&formatContext, RTSP_INPUT_URL, NULL, &opts) != 0) {
        fprintf(stderr, "ERROR: Failed to open RTSP stream\n");
        av_dict_free(&opts);
        return -1;
    }
    av_dict_free(&opts);
    
    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to find stream information\n");
        return -1;
    }
    printf("✓ RTSP connected\n");
    
    // Find video stream
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            width = formatContext->streams[i]->codecpar->width;
            height = formatContext->streams[i]->codecpar->height;
            vir_width = RK_ALIGN_16(width);
            vir_height = height;
            printf("✓ Video: %dx%d (stride: %d)\n", width, height, vir_width);
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
    
    sws_ctx = sws_getContext(
        codecContext->width, codecContext->height, codecContext->pix_fmt,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) {
        fprintf(stderr, "ERROR: Failed to create SWS context\n");
        return -1;
    }
    printf("✓ Decoder initialized\n");
    
    // Allocate frames
    avframe = av_frame_alloc();
    i420_frame = av_frame_alloc();
    packet = av_packet_alloc();
    if (!avframe || !i420_frame || !packet) {
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
    
    // Initialize MPI system
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        fprintf(stderr, "ERROR: RK_MPI_SYS_Init failed\n");
        return -1;
    }
    
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
    src_Blk = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_TRUE);
    data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    
    if (src_Pool == MB_INVALID_POOLID || src_Blk == MB_INVALID_HANDLE || !data) {
        fprintf(stderr, "ERROR: Memory pool creation failed\n");
        return -1;
    }
    printf("✓ Memory pool created\n");
    
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
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsplive || !g_rtsp_session) {
        fprintf(stderr, "ERROR: RTSP server initialization failed\n");
        return -1;
    }
    
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("✓ RTSP server: rtsp://<device_ip>:554/live/0\n");
    
    stRecvParam.s32RecvPicNum = -1;
    s32Ret = RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to start VENC: 0x%x\n", s32Ret);
        return -1;
    }
    
    // Start streaming thread
    pthread_t stream_thread;
    if (pthread_create(&stream_thread, NULL, GetMediaBuffer, NULL) != 0) {
        fprintf(stderr, "ERROR: Failed to create streaming thread\n");
        return -1;
    }
    
    // Allocate processing buffers
    temp_i420_buffer = (unsigned char *)malloc(width * height * 3 / 2);
    bgr_frame = new cv::Mat(height, width, CV_8UC3);
    if (!temp_i420_buffer || !bgr_frame) {
        fprintf(stderr, "ERROR: Failed to allocate processing buffers\n");
        return -1;
    }
    
    printf("\n========================================\n");
    printf("Processing started. Press Ctrl+C to stop.\n");
    printf("========================================\n\n");
    
    start_time = get_time_us();
    last_fps_time = start_time;
    
    // *** MAIN PROCESSING LOOP ***
    while (g_running && av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecContext, packet) == 0) {
                while (avcodec_receive_frame(codecContext, avframe) == 0) {
                    if (!data) continue;
                    
                    // 1. Decode RTSP to I420
                    sws_scale(sws_ctx,
                              (const uint8_t * const*)avframe->data, 
                              avframe->linesize,
                              0, codecContext->height,
                              i420_frame->data, i420_frame->linesize);
                    
                    // 2. Copy I420 to VENC buffer (always needed for display)
                    for (int i = 0; i < height; i++) {
                        memcpy(data + i * vir_width, 
                               i420_frame->data[0] + i * i420_frame->linesize[0], width);
                    }
                    
                    int venc_u_offset = vir_width * vir_height;
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(data + venc_u_offset + i * (vir_width / 2), 
                               i420_frame->data[1] + i * i420_frame->linesize[1], width / 2);
                    }
                    
                    int venc_v_offset = venc_u_offset + (vir_width * vir_height / 4);
                    for (int i = 0; i < height / 2; i++) {
                        memcpy(data + venc_v_offset + i * (vir_width / 2), 
                               i420_frame->data[2] + i * i420_frame->linesize[2], width / 2);
                    }
                    
                    // 3. AI processing (every 3rd frame for performance)
                    static int ai_frame_counter = 0;
                    ai_frame_counter++;
                    bool run_inference = (ai_frame_counter % 3 == 0);
                    
                    uint64_t ai_start = get_time_us();
                    
                    if (run_inference) {
                        // Copy to AI buffer (remove stride)
                        for (int i = 0; i < height; i++) {
                            memcpy(temp_i420_buffer + i * width, data + i * vir_width, width);
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
                        
                        // Convert to RGB for YOLOv5
                        cv::Mat i420_mat(height + height/2, width, CV_8UC1, temp_i420_buffer);
                        cv::cvtColor(i420_mat, *bgr_frame, cv::COLOR_YUV2RGB_I420);
                        letterbox(*bgr_frame, letterbox_output, width, height);
                        
                        // RKNN inference
                        memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letterbox_output.data, 
                               MODEL_WIDTH * MODEL_HEIGHT * 3);
                        memset(&od_results, 0, sizeof(object_detect_result_list));
                        
                        int ret = inference_yolov5_model(&rknn_app_ctx, &od_results);
                        
                        // Update tracker with new detections
                        if (ret == 0 && g_tracker) {
                            std::vector<Object> tracker_objects = 
                                convert_detections_to_tracker(od_results, width, height);
                            
                            if (!tracker_objects.empty()) {
                                g_active_tracks = g_tracker->update(tracker_objects);
                                
                                // Map track IDs to class labels
                                for (const auto& track : g_active_tracks) {
                                    if (track.tlwh.size() < 4) continue;
                                    
                                    cv::Rect_<float> track_rect(track.tlwh[0], track.tlwh[1], 
                                                                track.tlwh[2], track.tlwh[3]);
                                    
                                    float best_overlap = 0.0f;
                                    int best_label = -1;
                                    
                                    for (const auto& obj : tracker_objects) {
                                        cv::Rect_<float> intersection = track_rect & obj.rect;
                                        cv::Rect_<float> union_rect = track_rect | obj.rect;
                                        
                                        if (union_rect.area() > 0) {
                                            float overlap = intersection.area() / union_rect.area();
                                            if (overlap > best_overlap) {
                                                best_overlap = overlap;
                                                best_label = obj.label;
                                            }
                                        }
                                    }
                                    
                                    if (best_overlap > 0.3f && best_label >= 0) {
                                        g_track_labels[track.track_id] = best_label;
                                    }
                                }
                                
                                cleanup_stale_labels(g_active_tracks);
                            }
                        }
                    }
                    // On skip frames, g_active_tracks persists from last inference
                    // This creates smooth anti-flicker display
                    
                    uint64_t ai_time = get_time_us() - ai_start;
                    
                    // 4. OSD rendering (every frame using persistent g_active_tracks)
                    uint64_t osd_start = get_time_us();
                    
                    for (const auto& track : g_active_tracks) {
                        if (track.tlwh.size() < 4) continue;
                        
                        int sX = static_cast<int>(track.tlwh[0]);
                        int sY = static_cast<int>(track.tlwh[1]);
                        int bbox_w = static_cast<int>(track.tlwh[2]);
                        int bbox_h = static_cast<int>(track.tlwh[3]);
                        
                        if (sX < 0 || sY < 0 || bbox_w <= 10 || bbox_h <= 10) continue;
                        if (sX + bbox_w > width || sY + bbox_h > height) continue;
                        
                        draw_bbox_i420_correct(data, width, height, vir_width,
                                              sX, sY, bbox_w, bbox_h, 2);
                        
                        char label[64];
                        auto label_it = g_track_labels.find(track.track_id);
                        
                        if (label_it != g_track_labels.end() && 
                            label_it->second >= 0 && label_it->second < 80) {
                            snprintf(label, sizeof(label), "ID:%d %s %.0f%%", 
                                    track.track_id,
                                    coco_cls_to_name(label_it->second),
                                    track.score * 100);
                        } else {
                            snprintf(label, sizeof(label), "ID:%d %.0f%%", 
                                    track.track_id, track.score * 100);
                        }
                        
                        draw_text_labels_hybrid(data, width, height, vir_width, sX, sY, label);
                    }
                    
                    uint64_t osd_time = get_time_us() - osd_start;
                    
                    // 5. Send to encoder
                    RK_MPI_SYS_MmzFlushCache(src_Blk, RK_FALSE);
                    h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
                    h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
                    
                    s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
                    if (s32Ret != RK_SUCCESS) {
                        consecutive_errors++;
                        if (consecutive_errors > 10) {
                            printf("ERROR: Too many VENC errors\n");
                            g_running = false;
                            break;
                        }
                        continue;
                    }
                    consecutive_errors = 0;
                    
                    // 6. Performance monitoring
                    frame_count++;
                    fps_frame_count++;
                    
                    uint64_t current_time = get_time_us();
                    if (current_time - last_fps_time >= 1000000) {
                        double fps = (double)fps_frame_count / 
                                    ((current_time - last_fps_time) / 1000000.0);
                        double total_elapsed = (current_time - start_time) / 1000000.0;
                        
                        printf("[%.1fs] FPS: %.2f | AI: %.1fms (%s) | OSD: %.1fms | Tracks: %lu\n",
                               total_elapsed, fps, ai_time / 1000.0,
                               run_inference ? "RUN" : "SKIP",
                               osd_time / 1000.0, 
                               (unsigned long)g_active_tracks.size());
                        
                        fps_frame_count = 0;
                        last_fps_time = current_time;
                    }
                }
            }
        }
        av_packet_unref(packet);
    }
    
    // Performance summary
    uint64_t end_time = get_time_us();
    double total_time = (end_time - start_time) / 1000000.0;
    double avg_fps = total_time > 0 ? (double)frame_count / total_time : 0;
    
    printf("\n========================================\n");
    printf("Performance Summary\n");
    printf("========================================\n");
    printf("  Frames processed: %lu\n", (unsigned long)frame_count);
    printf("  Runtime: %.2f seconds\n", total_time);
    printf("  Average FPS: %.2f\n", avg_fps);
    printf("  Performance gain: %.1fx vs baseline\n", avg_fps / 9.5);
    printf("========================================\n");
    
    // Cleanup
    printf("Cleaning up...\n");
    
    g_running = false;
    pthread_join(stream_thread, NULL);
    
    if (g_tracker) delete g_tracker;
    g_active_tracks.clear();
    g_track_labels.clear();
    
    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);
    
    if (bgr_frame) delete bgr_frame;
    if (temp_i420_buffer) free(temp_i420_buffer);
    
    if (packet) av_packet_free(&packet);
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