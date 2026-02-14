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
#include <mutex>

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

#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"
#include <rga.h>
#include <im2d.h>
#include <im2d_type.h>
#include "RgaUtils.h"

// ============================================================================
// Macros & Constants
// ============================================================================
#define RK_ALIGN_16(x) (((x) + 15) & (~15))
#define RK_ALIGN_4(x)  (((x) + 3) & (~3))
#define RK_ALIGN_2(x)  (((x) + 1) & (~1))
#define MODEL_WIDTH  640
#define MODEL_HEIGHT 640
#define RTSP_INPUT_URL "rtsp://220.254.72.200/Src/MediaInput/h264/stream_1"

// ============================================================================
// Global state
// ============================================================================
static volatile bool g_running = true;
static float g_scale = 1.0f;
static int g_leftPadding = 0;
static int g_topPadding = 0;

rknn_app_context_t rknn_app_ctx;
object_detect_result_list od_results;
rtsp_demo_handle g_rtsplive = NULL;
rtsp_session_handle g_rtsp_session = NULL;

static BYTETracker* g_tracker = nullptr;
static std::vector<STrack> g_active_tracks;
static std::map<int, int> g_track_labels;

// ============================================================================
// RGA state
// ============================================================================
static bool g_rga_available = false;
static std::mutex g_rga_mutex;

static MB_POOL g_rga_dst_pool = MB_INVALID_POOLID;
static MB_BLK g_rga_dst_blk = MB_INVALID_HANDLE; 
static unsigned char* g_rga_dst_vir = nullptr;

static int g_scaled_w = 0;
static int g_scaled_h = 0;
static int g_src_vir_w = 0;
static int g_src_vir_h = 0;

// ============================================================================
// Init RGA
// ============================================================================
bool init_rga_acceleration(int width, int height, int vir_width, int vir_height) {
    g_src_vir_w = (vir_width  > width)  ? vir_width  : RK_ALIGN_16(width);
    g_src_vir_h = (vir_height > height) ? vir_height : RK_ALIGN_16(height);

    float scaleX = (float)MODEL_WIDTH / (float)width;
    float scaleY = (float)MODEL_HEIGHT / (float)height;
    g_scale = std::min(scaleX, scaleY);

    g_scaled_w = RK_ALIGN_4((int)((float)width * g_scale));
    g_scaled_h = RK_ALIGN_4((int)((float)height * g_scale));
    
    g_scaled_w = std::min(g_scaled_w, MODEL_WIDTH);
    g_scaled_h = std::min(g_scaled_h, MODEL_HEIGHT);

    g_leftPadding = (MODEL_WIDTH - g_scaled_w) / 2;
    g_topPadding  = (MODEL_HEIGHT - g_scaled_h) / 2;

    RK_U32 dst_size = MODEL_WIDTH * MODEL_HEIGHT * 3;
    MB_POOL_CONFIG_S pool_cfg;
    memset(&pool_cfg, 0, sizeof(pool_cfg));
    pool_cfg.u64MBSize   = dst_size;
    pool_cfg.u32MBCnt    = 1;
    pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA;
    pool_cfg.bPreAlloc   = RK_TRUE; 

    g_rga_dst_pool = RK_MPI_MB_CreatePool(&pool_cfg);
    if (g_rga_dst_pool == MB_INVALID_POOLID) return false;

    // Use Uncached for AI input consistency
    g_rga_dst_blk = RK_MPI_MB_GetMB(g_rga_dst_pool, dst_size, RK_FALSE);
    if (g_rga_dst_blk == MB_INVALID_HANDLE) return false;

    g_rga_dst_vir = (unsigned char*)RK_MPI_MB_Handle2VirAddr(g_rga_dst_blk);
    g_rga_available = true;
    return true;
}

void cleanup_rga_acceleration() {
    std::lock_guard<std::mutex> lock(g_rga_mutex);
    if (g_rga_dst_blk != MB_INVALID_HANDLE) RK_MPI_MB_ReleaseMB(g_rga_dst_blk);
    if (g_rga_dst_pool != MB_INVALID_POOLID) RK_MPI_MB_DestroyPool(g_rga_dst_pool);
    g_rga_available = false;
}

// ============================================================================
// RGA Process (I420 -> RGB)
// ============================================================================
uint64_t rga_preprocess(MB_BLK src_blk, int src_width, int src_height, 
                        int vir_width, int vir_height, unsigned char* output_npu_ptr) {
    if (!g_rga_available || src_blk == MB_INVALID_HANDLE) return 0;

    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    int src_fd = RK_MPI_MB_Handle2Fd(src_blk);
    int dst_fd = RK_MPI_MB_Handle2Fd(g_rga_dst_blk);
    if (src_fd < 0 || dst_fd < 0) return 0;

    rga_buffer_t src_buf = wrapbuffer_fd_t(src_fd, src_width, src_height, 
                                           vir_width, vir_height, RK_FORMAT_YCbCr_420_P);
    rga_buffer_t dst_buf = wrapbuffer_fd_t(dst_fd, g_scaled_w, g_scaled_h, 
                                           MODEL_WIDTH, g_scaled_h, RK_FORMAT_RGB_888);

    IM_STATUS ret = imresize(src_buf, dst_buf);
    if (ret != IM_STATUS_SUCCESS) {
         ret = imcvtcolor(src_buf, dst_buf, src_buf.format, dst_buf.format, IM_YUV_TO_RGB_BT601_LIMIT);
    }
    if (ret != IM_STATUS_SUCCESS) return 0;

    RK_MPI_SYS_MmzFlushCache(g_rga_dst_blk, RK_TRUE); 

    memset(output_npu_ptr, 0, MODEL_WIDTH * MODEL_HEIGHT * 3);
    for (int h = 0; h < g_scaled_h; h++) {
        unsigned char* src_ptr = g_rga_dst_vir + (h * MODEL_WIDTH * 3); 
        unsigned char* dst_ptr = output_npu_ptr + ((h + g_topPadding) * MODEL_WIDTH * 3) + (g_leftPadding * 3);
        memcpy(dst_ptr, src_ptr, g_scaled_w * 3);
    }

    gettimeofday(&tv1, NULL);
    return (uint64_t)(tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec);
}

// ============================================================================
// Fallback / Utils
// ============================================================================
void letterbox_cpu(cv::Mat& input, cv::Mat& output, int src_width, int src_height) {
    float scaleX = (float)MODEL_WIDTH / (float)src_width;
    float scaleY = (float)MODEL_HEIGHT / (float)src_height;
    g_scale = std::min(scaleX, scaleY);
    int inputWidth  = (int)((float)src_width * g_scale);
    int inputHeight = (int)((float)src_height * g_scale);
    g_leftPadding = (MODEL_WIDTH - inputWidth) / 2;
    g_topPadding  = (MODEL_HEIGHT - inputHeight) / 2;
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

void cleanup_stale_labels(const std::vector<STrack>& active_tracks) {
    std::set<int> active_ids;
    for (const auto& track : active_tracks) active_ids.insert(track.track_id);
    for (auto it = g_track_labels.begin(); it != g_track_labels.end();) {
        if (active_ids.find(it->first) == active_ids.end()) it = g_track_labels.erase(it);
        else ++it;
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

// ============================================================================
// OSD: PIXEL-BY-PIXEL SAFE DRAWING
// ============================================================================
void draw_text_labels_hybrid(unsigned char *yuv_data, int width, int height,
                             int vir_width, int x, int y, const char *text) {
    if (!text || strlen(text) == 0) return;

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int padding = 4;
    int label_w = (text_size.width + padding * 2);
    int label_h = (text_size.height + padding * 2);
    
    label_w = RK_ALIGN_2(label_w);

    x = std::max(0, std::min(x, width - label_w));
    y = std::max(0, std::min(y, height - label_h));
    int draw_y = (y > label_h + 4) ? (y - label_h - 2) : (y + 20);
    draw_y = std::max(0, std::min(draw_y, height - label_h));

    // 1. Draw Text to Grayscale Mask
    cv::Mat mask(label_h, label_w, CV_8UC1, cv::Scalar(0)); 
    cv::putText(mask, text, cv::Point(padding, label_h - padding),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1, cv::LINE_AA);

    // 2. Safe Pixel Copy Loop
    unsigned char *y_ptr = yuv_data;
    for (int row = 0; row < label_h; row++) {
        if (draw_y + row >= height) break;
        const uint8_t* src_row = mask.ptr<uint8_t>(row);
        uint8_t* dst_row = y_ptr + (draw_y + row) * vir_width + x;
        for(int col = 0; col < label_w; col++) {
            if(x + col >= width) break;
            dst_row[col] = src_row[col];
        }
    }

    // 3. Clear UV (I420 planar)
    unsigned char *u_ptr = yuv_data + vir_width * height;
    unsigned char *v_ptr = u_ptr + (vir_width * height / 4);
    
    int uv_x = x / 2;
    int uv_y = draw_y / 2;
    int uv_w = label_w / 2;
    int uv_h = label_h / 2;
    int uv_stride = vir_width / 2;

    for (int row = 0; row < uv_h; row++) {
        if (uv_y + row >= height/2) break;
        int uv_offset = (uv_y + row) * uv_stride + uv_x;
        memset(u_ptr + uv_offset, 128, uv_w);
        memset(v_ptr + uv_offset, 128, uv_w);
    }
}

void draw_bbox_i420_correct(unsigned char *data, int width, int height, int vir_width,
                           int x, int y, int w, int h, int thickness) {
    if (!data || x >= width || y >= height) return;
    w = std::min(w, width - x);
    h = std::min(h, height - y);
    if (w < 4 || h < 4) return;
    
    const uint8_t Y_VAL = 200, U_VAL = 44, V_VAL = 21;
    unsigned char *y_plane = data;
    unsigned char *u_plane = data + vir_width * height;
    unsigned char *v_plane = u_plane + (vir_width * height / 4);
    
    // Y Plane
    for (int t = 0; t < thickness; t++) {
        if (y+t < height) memset(y_plane + (y+t)*vir_width + x, Y_VAL, w);
        if (y+h-1-t >= 0 && y+h-1-t < height) memset(y_plane + (y+h-1-t)*vir_width + x, Y_VAL, w);
    }
    for (int r = y; r < y+h && r < height; r++) {
        for (int t = 0; t < thickness; t++) {
            if (x+t < width) y_plane[r*vir_width + x+t] = Y_VAL;
            if (x+w-1-t < width) y_plane[r*vir_width + x+w-1-t] = Y_VAL;
        }
    }
    
    // UV Plane
    int uv_stride = vir_width / 2;
    int uv_x = x/2, uv_y = y/2, uv_w = w/2, uv_h = h/2;
    int uv_thick = std::max(1, thickness/2);
    
    auto fill_uv = [&](unsigned char* plane, uint8_t val) {
        for (int t = 0; t < uv_thick; t++) {
            if (uv_y+t < height/2) memset(plane + (uv_y+t)*uv_stride + uv_x, val, uv_w);
            if (uv_y+uv_h-1-t >= 0 && uv_y+uv_h-1-t < height/2) memset(plane + (uv_y+uv_h-1-t)*uv_stride + uv_x, val, uv_w);
        }
        for (int r = uv_y; r < uv_y+uv_h && r < height/2; r++) {
            for (int t = 0; t < uv_thick; t++) {
                if (uv_x+t < uv_stride) plane[r*uv_stride + uv_x+t] = val;
                if (uv_x+uv_w-1-t < uv_stride) plane[r*uv_stride + uv_x+uv_w-1-t] = val;
            }
        }
    };
    fill_uv(u_plane, U_VAL);
    fill_uv(v_plane, V_VAL);
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
                rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);
                rtsp_do_event(g_rtsplive);
            }
            RK_MPI_VENC_ReleaseStream(0, &stFrame);
        }
        usleep(10000);
    }
    free(stFrame.pstPack);
    return NULL;
}

std::vector<Object> convert_detections_to_tracker(const object_detect_result_list& od_results, int width, int height) {
    std::vector<Object> objects;
    for (int i = 0; i < od_results.count; i++) {
        const auto& det = od_results.results[i];
        if (det.prop < 0.20f) continue;
        if (det.cls_id < 0 || det.cls_id >= 80) continue;
        int sX = (int)det.box.left; int sY = (int)det.box.top;
        int eX = (int)det.box.right; int eY = (int)det.box.bottom;
        mapCoordinates(&sX, &sY); mapCoordinates(&eX, &eY);
        sX = std::max(0, std::min(sX, width-1)); sY = std::max(0, std::min(sY, height-1));
        eX = std::max(0, std::min(eX, width-1)); eY = std::max(0, std::min(eY, height-1));
        if (eX-sX <= 8 || eY-sY <= 8) continue;
        Object obj;
        obj.rect = cv::Rect_<float>(sX, sY, eX-sX, eY-sY);
        obj.label = det.cls_id; obj.prob = det.prop;
        objects.push_back(obj);
    }
    return objects;
}

int main(int argc, char *argv[]) {
    system("RkLunch-stop.sh");
    signal(SIGINT, signal_handler);
    
    RK_S32 s32Ret = 0;
    cv::Mat letterbox_output(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
    const char *model_path = "./model/yolov5.rknn";
    unsigned char *temp_i420_buffer = NULL;
    cv::Mat *bgr_frame = NULL;
    VIDEO_FRAME_INFO_S h264_frame;
    VENC_RECV_PIC_PARAM_S stRecvParam;
    RK_U32 H264_TimeRef = 0;

    AVFormatContext *fmt_ctx = NULL;
    AVCodecContext *dec_ctx = NULL;
    AVFrame *avframe = NULL, *i420_frame = NULL;
    AVPacket *pkt = NULL;
    struct SwsContext *sws_ctx = NULL;
    int video_idx = -1;
    AVDictionary *opts = NULL;

    int width = 0, height = 0, vir_width = 0, vir_height = 0;
    MB_POOL src_Pool = MB_INVALID_POOLID;
    
    if (init_yolov5_model(model_path, &rknn_app_ctx) != 0 || init_post_process() != 0) return -1;
    g_tracker = new BYTETracker(5, 15);

    avformat_network_init();
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    if (avformat_open_input(&fmt_ctx, RTSP_INPUT_URL, NULL, &opts) != 0) return -1;
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) return -1;
    
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = i;
            width = fmt_ctx->streams[i]->codecpar->width;
            height = fmt_ctx->streams[i]->codecpar->height;
            vir_width = RK_ALIGN_16(width);
            vir_height = height;
            break;
        }
    }
    
    AVCodec *codec = avcodec_find_decoder(fmt_ctx->streams[video_idx]->codecpar->codec_id);
    dec_ctx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[video_idx]->codecpar);
    avcodec_open2(dec_ctx, codec, NULL);

    sws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        width, height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);

    avframe = av_frame_alloc();
    i420_frame = av_frame_alloc();
    pkt = av_packet_alloc();
    i420_frame->format = AV_PIX_FMT_YUV420P;
    i420_frame->width = width;
    i420_frame->height = height;
    av_frame_get_buffer(i420_frame, 32);

    RK_MPI_SYS_Init();
    venc_init(0, width, height, RK_VIDEO_ID_AVC);

    // --- INCREASE BUFFER COUNT TO 6 FOR STABILITY ---
    MB_POOL_CONFIG_S PoolCfg;
    memset(&PoolCfg, 0, sizeof(PoolCfg));
    PoolCfg.u64MBSize = vir_width * vir_height * 3 / 2;
    PoolCfg.u32MBCnt = 6; 
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
    src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);

    init_rga_acceleration(width, height, vir_width, vir_height);

    g_rtsplive     = create_rtsp_demo(554);
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

    stRecvParam.s32RecvPicNum = -1;
    RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);

    pthread_t stream_thread;
    pthread_create(&stream_thread, NULL, GetMediaBuffer, NULL);

    temp_i420_buffer = (unsigned char *)malloc(width * height * 3 / 2);
    bgr_frame = new cv::Mat(height, width, CV_8UC3);

    memset(&h264_frame, 0, sizeof(h264_frame));
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = vir_width;
    h264_frame.stVFrame.u32VirHeight = vir_height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_YUV420P;
    h264_frame.stVFrame.u32FrameFlag = 160;

    int ai_counter = 0;
    
    // LOGGING VARIABLES
    uint64_t frame_count = 0;
    uint64_t fps_frame_count = 0;
    uint64_t start_time = get_time_us();
    uint64_t last_fps_time = start_time;
    
    printf("--- Running (Buffered I420) ---\n");

    while (g_running && av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == video_idx) {
            avcodec_send_packet(dec_ctx, pkt);
            while (avcodec_receive_frame(dec_ctx, avframe) == 0) {
                
                // 1. Get Fresh Buffer (Wait if pool is full, don't skip)
                MB_BLK mb = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_FALSE); 
                if (mb == MB_INVALID_HANDLE) {
                    usleep(1000); // Small wait to let VENC release
                    continue; 
                }
                
                unsigned char* data = (unsigned char*)RK_MPI_MB_Handle2VirAddr(mb);
                
                sws_scale(sws_ctx, (const uint8_t * const*)avframe->data, avframe->linesize,
                          0, dec_ctx->height, i420_frame->data, i420_frame->linesize);

                for (int i = 0; i < height; i++)
                    memcpy(data + i * vir_width, i420_frame->data[0] + i * i420_frame->linesize[0], width);
                
                int u_off = vir_width * vir_height;
                for (int i = 0; i < height / 2; i++)
                    memcpy(data + u_off + i * (vir_width/2), i420_frame->data[1] + i * i420_frame->linesize[1], width/2);
                
                int v_off = u_off + (vir_width * vir_height / 4);
                for (int i = 0; i < height / 2; i++)
                    memcpy(data + v_off + i * (vir_width/2), i420_frame->data[2] + i * i420_frame->linesize[2], width/2);

                ai_counter++;
                bool run_inference = (ai_counter % 2 == 0); 
                bool used_rga = false;
                uint64_t ai_time = 0;
                uint64_t preprocess_time = 0;

                if (run_inference) {
                    uint64_t ai_start = get_time_us();
                    
                    preprocess_time = rga_preprocess(mb, width, height, vir_width, vir_height, 
                                      (unsigned char*)rknn_app_ctx.input_mems[0]->virt_addr);
                    used_rga = (preprocess_time > 0);
                    
                    if (!used_rga) {
                        uint64_t t0 = get_time_us();
                        for (int i = 0; i < height; i++) memcpy(temp_i420_buffer + i * width, data + i * vir_width, width);
                        // fallback logic would go here...
                        preprocess_time = get_time_us() - t0;
                    }

                    memset(&od_results, 0, sizeof(od_results));
                    if (inference_yolov5_model(&rknn_app_ctx, &od_results) == 0 && g_tracker) {
                        std::vector<Object> objs = convert_detections_to_tracker(od_results, width, height);
                        if (!objs.empty()) {
                            g_active_tracks = g_tracker->update(objs);
                            for (const auto& tr : g_active_tracks) {
                                if (tr.tlwh.size() < 4) continue;
                                cv::Rect_<float> r(tr.tlwh[0], tr.tlwh[1], tr.tlwh[2], tr.tlwh[3]);
                                float max_ov = 0; int lbl = -1;
                                for (const auto& o : objs) {
                                    float ov = (r & o.rect).area() / (r | o.rect).area();
                                    if (ov > max_ov) { max_ov = ov; lbl = o.label; }
                                }
                                if (max_ov > 0.3f) g_track_labels[tr.track_id] = lbl;
                            }
                            cleanup_stale_labels(g_active_tracks);
                        }
                    }
                    ai_time = get_time_us() - ai_start;
                }

                // 4. OSD (Safe Loop + Flush)
                uint64_t osd_start = get_time_us();
                for (const auto& t : g_active_tracks) {
                    int sX = (int)t.tlwh[0], sY = (int)t.tlwh[1], w = (int)t.tlwh[2], h = (int)t.tlwh[3];
                    draw_bbox_i420_correct(data, width, height, vir_width, sX, sY, w, h, 2);
                    char lbl[64];
                    auto it = g_track_labels.find(t.track_id);
                    snprintf(lbl, sizeof(lbl), "ID:%d %s", t.track_id, 
                             (it != g_track_labels.end()) ? coco_cls_to_name(it->second) : "");
                    draw_text_labels_hybrid(data, width, height, vir_width, sX, sY, lbl);
                }
                uint64_t osd_time = get_time_us() - osd_start;

                // CRITICAL FIX: Flush Write-Buffer to RAM so VENC sees OSD
                RK_MPI_SYS_MmzFlushCache(mb, RK_FALSE);

                h264_frame.stVFrame.pMbBlk = mb; 
                h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
                h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
                
                RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
                RK_MPI_MB_ReleaseMB(mb); 

                // LOGGING
                frame_count++;
                fps_frame_count++;
                uint64_t now = get_time_us();
                if (now - last_fps_time >= 1000000) {
                    double fps = (double)fps_frame_count / ((now - last_fps_time) / 1000000.0);
                    double elapsed = (now - start_time) / 1000000.0;
                    if (run_inference) {
                        printf("[%6.1fs] FPS:%5.1f | AI:%5.1fms | Pre:%5.1fms (%s) | OSD:%4.1fms\n",
                               elapsed, fps, ai_time / 1000.0, preprocess_time / 1000.0, 
                               used_rga ? "RGA" : "CPU", osd_time / 1000.0);
                    } else {
                        printf("[%6.1fs] FPS:%5.1f | AI: SKIP | OSD:%4.1fms\n",
                               elapsed, fps, osd_time / 1000.0);
                    }
                    fps_frame_count = 0;
                    last_fps_time = now;
                }
            }
        }
        av_packet_unref(pkt);
    }

    if (g_tracker) delete g_tracker;
    if (bgr_frame) delete bgr_frame;
    if (temp_i420_buffer) free(temp_i420_buffer);
    if (pkt) av_packet_free(&pkt);
    if (i420_frame) av_frame_free(&i420_frame);
    if (avframe) av_frame_free(&avframe);
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (dec_ctx) avcodec_free_context(&dec_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
    avformat_network_deinit();
    if (src_Pool != MB_INVALID_POOLID) RK_MPI_MB_DestroyPool(src_Pool);
    if (g_rtsplive) rtsp_del_demo(g_rtsplive);
    RK_MPI_SYS_Exit();
    cleanup_rga_acceleration();
    release_yolov5_model(&rknn_app_ctx);
    deinit_post_process();
    return 0;
}