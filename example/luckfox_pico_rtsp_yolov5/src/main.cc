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
// Macros
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
// RGA state (Fixed for RV1106 DMA)
// ============================================================================
static bool g_rga_available = false;
static std::mutex g_rga_mutex;

// We use MB_BLK instead of raw pointers for RGA buffers to ensure physical continuity
static MB_POOL g_rga_dst_pool = MB_INVALID_POOLID; // Dedicated pool for RGA output
static MB_BLK g_rga_dst_blk = MB_INVALID_HANDLE; 
static unsigned char* g_rga_dst_vir = nullptr; // Mapped virtual address for CPU reading

static int g_scaled_w = 0;
static int g_scaled_h = 0;
static int g_src_vir_w = 0;
static int g_src_vir_h = 0;

// ============================================================================
// RGA init / cleanup / utils
// ============================================================================

bool init_rga_acceleration(int width, int height,
                           int vir_width, int vir_height)
{
    g_src_vir_w = (vir_width  > width)  ? vir_width  : RK_ALIGN_16(width);
    g_src_vir_h = (vir_height > height) ? vir_height : RK_ALIGN_16(height);

    // Calculate scale factors
    float scaleX = (float)MODEL_WIDTH / (float)width;
    float scaleY = (float)MODEL_HEIGHT / (float)height;
    g_scale = std::min(scaleX, scaleY);

    g_scaled_w = RK_ALIGN_4((int)((float)width * g_scale));
    g_scaled_h = RK_ALIGN_2((int)((float)height * g_scale));
    
    // Clamp to model size
    g_scaled_w = std::min(g_scaled_w, MODEL_WIDTH);
    g_scaled_h = std::min(g_scaled_h, MODEL_HEIGHT);

    g_leftPadding = (MODEL_WIDTH - g_scaled_w) / 2;
    g_topPadding  = (MODEL_HEIGHT - g_scaled_h) / 2;

    // --- ALLOCATION FIX ---
    // Allocate ONE buffer for the resized RGB image using MPI (DMA memory)
    // Size = Model Width * Model Height * 3 (RGB888)
    RK_U32 dst_size = MODEL_WIDTH * MODEL_HEIGHT * 3;
    MB_POOL_CONFIG_S pool_cfg;
    memset(&pool_cfg, 0, sizeof(pool_cfg));
    pool_cfg.u64MBSize   = dst_size;
    pool_cfg.u32MBCnt    = 1;  // We only need 1 buffer for RGA output
    pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA; // Allocate from DMA/CMA
    pool_cfg.bPreAlloc   = RK_TRUE; 

    // 1. Create the pool
    g_rga_dst_pool = RK_MPI_MB_CreatePool(&pool_cfg);
    if (g_rga_dst_pool == MB_INVALID_POOLID) {
        printf("ERROR: Failed to create RGA destination pool (size: %d)\n", dst_size);
        return false;
    }

    // 2. Get the block from the pool
    g_rga_dst_blk = RK_MPI_MB_GetMB(g_rga_dst_pool, dst_size, RK_TRUE);
    if (g_rga_dst_blk == MB_INVALID_HANDLE) {
        printf("ERROR: Failed to get MB from RGA pool\n");
        RK_MPI_MB_DestroyPool(g_rga_dst_pool);
        g_rga_dst_pool = MB_INVALID_POOLID;
        return false;
    }

    // 3. Get the virtual address
    g_rga_dst_vir = (unsigned char*)RK_MPI_MB_Handle2VirAddr(g_rga_dst_blk);

    g_rga_available = true;
    printf("  RGA Init: Src %dx%d -> Dst %dx%d (Scale: %.2f)\n", 
           width, height, g_scaled_w, g_scaled_h, g_scale);
    printf("  RGA Memory: Allocated %d bytes via dedicated MPI pool\n", dst_size);
    printf("  RGA Version: %s\n", querystring(RGA_VERSION));

    return true;
}

void cleanup_rga_acceleration() {
    std::lock_guard<std::mutex> lock(g_rga_mutex);
    if (g_rga_dst_blk != MB_INVALID_HANDLE) {
        RK_MPI_MB_ReleaseMB(g_rga_dst_blk);
        g_rga_dst_blk = MB_INVALID_HANDLE;
    }
    if (g_rga_dst_pool != MB_INVALID_POOLID) {
        RK_MPI_MB_DestroyPool(g_rga_dst_pool);
        g_rga_dst_pool = MB_INVALID_POOLID;
    }
    g_rga_dst_vir = nullptr;
    g_rga_available = false;
}

// ============================================================================
// RGA preprocessing — NV12 DMA -> Resize/CSC -> RGB DMA
// ============================================================================

uint64_t rga_preprocess(
    MB_BLK   src_blk,
    int      src_width,
    int      src_height,
    int      vir_width,
    int      vir_height,
    unsigned char* output_npu_ptr)
{
    if (!g_rga_available || src_blk == MB_INVALID_HANDLE || g_rga_dst_blk == MB_INVALID_HANDLE)
        return 0;

    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    std::lock_guard<std::mutex> lock(g_rga_mutex);

    // 1. Get File Descriptors (FD) for DMA hardware access
    // This prevents the "Current NONE_MMU[0] cannot support physicaly discontinuous virtual address" error
    int src_fd = RK_MPI_MB_Handle2Fd(src_blk);
    int dst_fd = RK_MPI_MB_Handle2Fd(g_rga_dst_blk);

    if (src_fd < 0 || dst_fd < 0) {
        printf("RGA Error: Invalid FDs (src=%d, dst=%d)\n", src_fd, dst_fd);
        return 0;
    }

    // 2. Wrap Source (NV12) using FD
    rga_buffer_t src_buf = wrapbuffer_fd_t(
        src_fd,
        src_width, src_height,
        vir_width, vir_height,
        RK_FORMAT_YCbCr_420_SP // NV12
    );

    // 3. Wrap Destination (RGB888) using FD
    // Note: We use the actual scaled dimensions, but stride is the Model Width
    // This allows us to write tightly packed lines that match the model width stride if needed,
    // or we handle stride manually in the CPU copy.
    rga_buffer_t dst_buf = wrapbuffer_fd_t(
        dst_fd,
        g_scaled_w, g_scaled_h,
        MODEL_WIDTH, g_scaled_h, // Stride = Model Width
        RK_FORMAT_RGB_888
    );

    // 4. Perform Resize + Color Conversion in one pass
    // IM_YUV_TO_RGB_BT601_LIMIT is standard for video
    IM_STATUS ret = imresize(src_buf, dst_buf);
    
    if (ret != IM_STATUS_SUCCESS) {
         printf("RGA Error: imresize failed: %s\n", imStrError(ret));
         return 0;
    }

    // 5. Letterbox Copy (DMA buffer -> NPU Input)
    // We already have the virtual pointer g_rga_dst_vir from initialization.
    // We must manually copy into the center of the NPU buffer to apply padding.
    
    // Clear NPU buffer (black padding)
    memset(output_npu_ptr, 0, MODEL_WIDTH * MODEL_HEIGHT * 3);

    // Copy the valid image area into the center of the NPU buffer
    for (int h = 0; h < g_scaled_h; h++) {
        // Source: Our RGA output buffer (Stride = MODEL_WIDTH)
        unsigned char* src_ptr = g_rga_dst_vir + (h * MODEL_WIDTH * 3); 
        
        // Dest: The NPU input buffer
        unsigned char* dst_ptr = output_npu_ptr + 
                                 ((h + g_topPadding) * MODEL_WIDTH * 3) + 
                                 (g_leftPadding * 3);
        
        memcpy(dst_ptr, src_ptr, g_scaled_w * 3);
    }

    gettimeofday(&tv1, NULL);
    return (uint64_t)(tv1.tv_sec - tv0.tv_sec) * 1000000 +
           (tv1.tv_usec - tv0.tv_usec);
}

// ============================================================================
// CPU fallback letterbox (unchanged)
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
    cv::resize(input, inputScale, cv::Size(inputWidth, inputHeight),
               0, 0, cv::INTER_NEAREST);

    output.setTo(cv::Scalar(0, 0, 0));
    cv::Rect roi(g_leftPadding, g_topPadding, inputWidth, inputHeight);
    inputScale.copyTo(output(roi));
}

// ============================================================================
// Coordinate mapping (unchanged)
// ============================================================================

void mapCoordinates(int *x, int *y) {
    int mx = *x - g_leftPadding;
    int my = *y - g_topPadding;
    *x = (int)((float)mx / g_scale);
    *y = (int)((float)my / g_scale);
}

// ============================================================================
// Utility functions (unchanged)
// ============================================================================

void cleanup_stale_labels(const std::vector<STrack>& active_tracks) {
    std::set<int> active_ids;
    for (const auto& track : active_tracks) active_ids.insert(track.track_id);
    for (auto it = g_track_labels.begin(); it != g_track_labels.end();) {
        if (active_ids.find(it->first) == active_ids.end())
            it = g_track_labels.erase(it);
        else
            ++it;
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
// OSD drawing — NV12 version (Y plane + interleaved UV plane)
// ============================================================================

void draw_text_labels_hybrid(unsigned char *yuv_data, int width, int height,
                             int vir_width, int x, int y, const char *text) {
    if (!text || strlen(text) == 0) return;

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 2, &baseline);
    int padding = 5;
    int label_w = text_size.width + padding * 2;
    int label_h = text_size.height + padding * 2;

    int text_x = x;
    int text_y = (y > label_h + 5) ? (y - label_h - 2) : (y + 20);

    text_x = std::max(0, std::min(text_x, width - label_w));
    text_y = std::max(0, std::min(text_y, height - label_h));

    cv::Mat label_img(label_h, label_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(label_img, text, cv::Point(padding, label_h - padding),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),
                2, cv::LINE_AA);

    unsigned char *y_plane  = yuv_data;
    // FIX: NV12 has interleaved UV as a single plane after Y
    unsigned char *uv_plane = yuv_data + vir_width * height;

    for (int row = 0; row < label_h && text_y + row < height; row++) {
        uint8_t *bgr = label_img.ptr<uint8_t>(row);
        int y_off = (text_y + row) * vir_width + text_x;
        for (int col = 0; col < label_w && text_x + col < width; col++) {
            uint8_t b = bgr[col * 3];
            uint8_t g = bgr[col * 3 + 1];
            uint8_t r = bgr[col * 3 + 2];
            y_plane[y_off + col] = (uint8_t)(0.299f*r + 0.587f*g + 0.114f*b);
        }
    }

    // NV12 UV: interleaved U,V pairs at half resolution
    int uv_x = (text_x / 2) * 2;  // align to pair boundary
    int uv_y_start = text_y / 2;
    int uv_y_end   = std::min((text_y + label_h) / 2, height / 2);
    int uv_w_bytes = ((label_w + 1) / 2) * 2;  // bytes = pairs * 2

    for (int r = uv_y_start; r < uv_y_end; r++) {
        // Set UV to 128,128 (neutral) for green-ish text on NV12
        memset(uv_plane + r * vir_width + uv_x, 128, uv_w_bytes);
    }
}

void draw_bbox_nv12(unsigned char *data, int width, int height,
                    int vir_width, int x, int y, int w, int h,
                    int thickness) {
    if (!data || thickness <= 0 || x < 0 || y < 0 || w <= 0 || h <= 0) return;
    if (x >= width || y >= height) return;

    w = std::min(w, width - x);
    h = std::min(h, height - y);
    if (w < 10 || h < 10) return;
    thickness = std::min(thickness, std::min(w/4, h/4));

    const uint8_t Y_GREEN = 150;

    unsigned char *y_plane  = data;
    unsigned char *uv_plane = data + vir_width * height;

    // Y plane: draw rectangle border
    for (int t = 0; t < thickness; t++) {
        if (y+t < height)
            memset(y_plane + (y+t)*vir_width + x, Y_GREEN, w);
        if (y+h-1-t >= 0 && y+h-1-t < height)
            memset(y_plane + (y+h-1-t)*vir_width + x, Y_GREEN, w);
    }
    for (int row = y; row < y+h && row < height; row++) {
        for (int t = 0; t < thickness; t++) {
            if (x+t < width) y_plane[row*vir_width + x+t] = Y_GREEN;
            if (x+w-1-t >= 0 && x+w-1-t < width)
                y_plane[row*vir_width + x+w-1-t] = Y_GREEN;
        }
    }

    // NV12 UV plane: interleaved at half res, stride = vir_width
    int uv_x = (x / 2) * 2;
    int uv_y = y / 2;
    int uv_w_bytes = std::min((w / 2) * 2, (int)(vir_width - uv_x));
    int uv_h = std::min(h / 2, height / 2 - uv_y);
    int uv_thickness = std::max(1, thickness / 2);

    // Green in YUV: U≈44, V≈21 → interleaved bytes: 44,21,44,21,...
    // For simplicity, approximate with U=44, V=21 pairs
    auto fill_uv_row = [&](int row_idx, int count) {
        unsigned char* p = uv_plane + row_idx * vir_width + uv_x;
        for (int i = 0; i < count && uv_x + i < vir_width; i += 2) {
            p[i]   = 44;   // U
            if (i+1 < count) p[i+1] = 21;  // V
        }
    };

    // Top/bottom edges
    for (int t = 0; t < uv_thickness; t++) {
        if (uv_y+t < height/2)
            fill_uv_row(uv_y+t, uv_w_bytes);
        if (uv_y+uv_h-1-t >= 0 && uv_y+uv_h-1-t < height/2)
            fill_uv_row(uv_y+uv_h-1-t, uv_w_bytes);
    }
    // Left/right edges
    for (int row = uv_y; row < uv_y+uv_h && row < height/2; row++) {
        for (int t = 0; t < uv_thickness; t++) {
            unsigned char* p = uv_plane + row * vir_width;
            int lx = uv_x + t*2;
            if (lx+1 < vir_width) { p[lx] = 44; p[lx+1] = 21; }
            int rx = uv_x + (uv_w_bytes - 2 - t*2);
            if (rx >= 0 && rx+1 < vir_width) { p[rx] = 44; p[rx+1] = 21; }
        }
    }
}

// ============================================================================
// VENC streaming thread (unchanged)
// ============================================================================

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

// ============================================================================
// Detection -> tracker conversion (unchanged)
// ============================================================================

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

        int bw = eX - sX, bh = eY - sY;
        if (bw <= 8 || bh <= 8) continue;

        Object obj;
        obj.rect = cv::Rect_<float>((float)sX, (float)sY, (float)bw, (float)bh);
        obj.label = det.cls_id;
        obj.prob = det.prop;
        objects.push_back(obj);
    }
    return objects;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char *argv[]) {
    system("RkLunch-stop.sh");
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    RK_S32 s32Ret = 0;
    cv::Mat letterbox_output(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);

    const char *model_path = "./model/yolov5.rknn";
    unsigned char *temp_nv12_buffer = NULL;
    cv::Mat *bgr_frame = NULL;

    VIDEO_FRAME_INFO_S h264_frame;
    VENC_RECV_PIC_PARAM_S stRecvParam;
    RK_U32 H264_TimeRef = 0;

    AVFormatContext *formatContext = NULL;
    AVCodecContext  *codecContext  = NULL;
    AVCodec         *codec        = NULL;
    AVPacket        *packet       = NULL;
    AVFrame         *avframe      = NULL;
    AVFrame         *nv12_frame   = NULL;
    struct SwsContext *sws_ctx    = NULL;
    int videoStreamIndex = -1;
    AVDictionary *opts = NULL;

    int width = 0, height = 0, vir_width = 0, vir_height = 0;

    MB_POOL src_Pool = MB_INVALID_POOLID;
    MB_BLK  src_Blk  = MB_INVALID_HANDLE;
    MB_POOL_CONFIG_S PoolCfg;
    unsigned char *data = NULL;

    uint64_t frame_count = 0, start_time = 0, last_fps_time = 0, fps_frame_count = 0;
    int consecutive_errors = 0;

    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&h264_frame,   0, sizeof(h264_frame));
    memset(&PoolCfg,      0, sizeof(PoolCfg));
    memset(&stRecvParam,  0, sizeof(stRecvParam));

    printf("========================================\n");
    printf("RTSP YOLOv5 + ByteTrack + RGA (NV12)\n");
    printf("========================================\n");

    // --- YOLOv5 ---
    if (init_yolov5_model(model_path, &rknn_app_ctx) != 0 || init_post_process() != 0) {
        fprintf(stderr, "ERROR: YOLOv5 init failed\n"); return -1;
    }
    printf("[OK] YOLOv5\n");

    // --- ByteTracker ---
    g_tracker = new BYTETracker(5, 15);
    if (!g_tracker) { fprintf(stderr, "ERROR: ByteTracker\n"); return -1; }
    printf("[OK] ByteTracker\n");

    // --- RTSP input ---
    avformat_network_init();
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay",     "500000", 0);
    av_dict_set(&opts, "stimeout",      "5000000", 0);
    av_dict_set(&opts, "buffer_size",   "1024000", 0);

    if (avformat_open_input(&formatContext, RTSP_INPUT_URL, NULL, &opts) != 0) {
        fprintf(stderr, "ERROR: RTSP open\n"); av_dict_free(&opts); return -1;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        fprintf(stderr, "ERROR: stream info\n"); return -1;
    }
    printf("[OK] RTSP connected\n");

    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            width      = formatContext->streams[i]->codecpar->width;
            height     = formatContext->streams[i]->codecpar->height;
            vir_width  = RK_ALIGN_16(width);
            vir_height = RK_ALIGN_16(height);
            printf("[OK] Video: %dx%d (stride %dx%d)\n",
                   width, height, vir_width, vir_height);
            break;
        }
    }
    if (videoStreamIndex == -1) { fprintf(stderr, "ERROR: no video\n"); return -1; }

    // --- Decoder ---
    codecContext = avcodec_alloc_context3(NULL);
    if (!codecContext) { fprintf(stderr, "ERROR: codec ctx\n"); return -1; }
    avcodec_parameters_to_context(codecContext,
        formatContext->streams[videoStreamIndex]->codecpar);
    codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec || avcodec_open2(codecContext, codec, NULL) < 0) {
        fprintf(stderr, "ERROR: decoder\n"); return -1;
    }

    // FIX: Output NV12 instead of I420 — eliminates three-plane RGA issue
    sws_ctx = sws_getContext(
        codecContext->width, codecContext->height, codecContext->pix_fmt,
        width, height, AV_PIX_FMT_NV12,     // <-- was AV_PIX_FMT_YUV420P
        SWS_FAST_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) { fprintf(stderr, "ERROR: sws\n"); return -1; }
    printf("[OK] Decoder (NV12 output)\n");

    avframe    = av_frame_alloc();
    nv12_frame = av_frame_alloc();
    packet     = av_packet_alloc();
    if (!avframe || !nv12_frame || !packet) {
        fprintf(stderr, "ERROR: alloc frames\n"); return -1;
    }
    nv12_frame->format = AV_PIX_FMT_NV12;
    nv12_frame->width  = width;
    nv12_frame->height = height;
    if (av_frame_get_buffer(nv12_frame, 32) < 0) {
        fprintf(stderr, "ERROR: nv12 buffer\n"); return -1;
    }

    // --- MPI / VENC ---
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        fprintf(stderr, "ERROR: MPI init\n"); return -1;
    }
    s32Ret = venc_init(0, width, height, RK_VIDEO_ID_AVC);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: VENC 0x%x\n", s32Ret); return -1;
    }
    printf("[OK] VENC\n");

    // --- Memory pool: NV12 = Y + UV = vir_w * vir_h * 3/2 ---
    PoolCfg.u64MBSize   = vir_width * vir_height * 3 / 2;
    PoolCfg.u32MBCnt    = 3;
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;

    src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    src_Blk  = RK_MPI_MB_GetMB(src_Pool, PoolCfg.u64MBSize, RK_TRUE);
    data     = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);

    if (src_Pool == MB_INVALID_POOLID || src_Blk == MB_INVALID_HANDLE || !data) {
        fprintf(stderr, "ERROR: pool\n"); return -1;
    }
    memset(data, 0, PoolCfg.u64MBSize);
    printf("[OK] Memory pool (%d bytes)\n", (int)PoolCfg.u64MBSize);

    // --- RGA init ---
    bool rga_enabled = init_rga_acceleration(width, height, vir_width, vir_height);
    if (!rga_enabled) {
        printf("[WARN] CPU-only mode\n");
    }

    // --- Frame structure: NV12 ---
    h264_frame.stVFrame.u32Width      = width;
    h264_frame.stVFrame.u32Height     = height;
    h264_frame.stVFrame.u32VirWidth   = vir_width;
    h264_frame.stVFrame.u32VirHeight  = vir_height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_YUV420SP;  // FIX: NV12
    h264_frame.stVFrame.u32FrameFlag  = 160;
    h264_frame.stVFrame.pMbBlk        = src_Blk;

    // --- RTSP output server ---
    g_rtsplive     = create_rtsp_demo(554);
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    if (!g_rtsplive || !g_rtsp_session) {
        fprintf(stderr, "ERROR: RTSP server\n"); return -1;
    }
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());
    printf("[OK] RTSP server: rtsp://<ip>:554/live/0\n");

    stRecvParam.s32RecvPicNum = -1;
    s32Ret = RK_MPI_VENC_StartRecvFrame(0, &stRecvParam);
    if (s32Ret != RK_SUCCESS) {
        fprintf(stderr, "ERROR: VENC start 0x%x\n", s32Ret); return -1;
    }

    pthread_t stream_thread;
    if (pthread_create(&stream_thread, NULL, GetMediaBuffer, NULL) != 0) {
        fprintf(stderr, "ERROR: thread\n"); return -1;
    }

    // CPU fallback buffers
    temp_nv12_buffer = (unsigned char *)malloc(width * height * 3 / 2);
    bgr_frame = new cv::Mat(height, width, CV_8UC3);
    if (!temp_nv12_buffer || !bgr_frame) {
        fprintf(stderr, "ERROR: cpu buffers\n"); return -1;
    }

    printf("\n========================================\n");
    printf("Running. Ctrl+C to stop.\n");
    printf("========================================\n\n");

    start_time   = get_time_us();
    last_fps_time = start_time;

    // ================================================================
    // MAIN LOOP
    // ================================================================
    while (g_running && av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index != videoStreamIndex) {
            av_packet_unref(packet);
            continue;
        }

        if (avcodec_send_packet(codecContext, packet) != 0) {
            av_packet_unref(packet);
            continue;
        }

        while (avcodec_receive_frame(codecContext, avframe) == 0) {
            if (!data) continue;

            // --------------------------------------------------------
            // Step 1: sws_scale — decoder output -> NV12
            // --------------------------------------------------------
            sws_scale(sws_ctx,
                      (const uint8_t * const*)avframe->data,
                      avframe->linesize,
                      0, codecContext->height,
                      nv12_frame->data, nv12_frame->linesize);

            // --------------------------------------------------------
            // Step 2: Copy NV12 into VENC DMA buffer (strided)
            //
            // NV12 layout in DMA buffer:
            //   Y  plane: vir_width * vir_height bytes  (offset 0)
            //   UV plane: vir_width * vir_height/2 bytes (interleaved U,V)
            // --------------------------------------------------------
            // Y plane
            for (int i = 0; i < height; i++)
                memcpy(data + i * vir_width,
                       nv12_frame->data[0] + i * nv12_frame->linesize[0],
                       width);

            // UV plane (interleaved, width bytes per row, height/2 rows)
            int uv_offset = vir_width * vir_height;
            for (int i = 0; i < height / 2; i++)
                memcpy(data + uv_offset + i * vir_width,
                       nv12_frame->data[1] + i * nv12_frame->linesize[1],
                       width);

            // --------------------------------------------------------
            // Step 3: AI inference (every 2nd frame)
            // --------------------------------------------------------
            static int ai_counter = 0;
            ai_counter++;
            bool run_inference = (ai_counter % 2 == 0);

            uint64_t ai_start = get_time_us();
            uint64_t preprocess_time = 0;
            bool used_rga = false;

            if (run_inference) {
                // RGA Preprocess now uses the DMA block (src_Blk) properly
                preprocess_time = rga_preprocess(
                    src_Blk, width, height, vir_width, vir_height,
                    (unsigned char*)rknn_app_ctx.input_mems[0]->virt_addr);

                if (preprocess_time > 0) {
                    used_rga = true;
                } else {
                    // CPU fallback: de-stride NV12, convert, letterbox
                    uint64_t cpu_t0 = get_time_us();

                    // De-stride Y
                    for (int i = 0; i < height; i++)
                        memcpy(temp_nv12_buffer + i * width,
                               data + i * vir_width, width);
                    // De-stride UV
                    int tu = width * height;
                    for (int i = 0; i < height / 2; i++)
                        memcpy(temp_nv12_buffer + tu + i * width,
                               data + uv_offset + i * vir_width, width);

                    cv::Mat nv12_mat(height + height/2, width, CV_8UC1,
                                    temp_nv12_buffer);
                    cv::cvtColor(nv12_mat, *bgr_frame, cv::COLOR_YUV2RGB_NV12);
                    letterbox_cpu(*bgr_frame, letterbox_output, width, height);
                    memcpy(rknn_app_ctx.input_mems[0]->virt_addr,
                           letterbox_output.data,
                           MODEL_WIDTH * MODEL_HEIGHT * 3);

                    preprocess_time = get_time_us() - cpu_t0;
                }

                memset(&od_results, 0, sizeof(od_results));
                int ret = inference_yolov5_model(&rknn_app_ctx, &od_results);

                if (ret == 0 && g_tracker) {
                    std::vector<Object> objs =
                        convert_detections_to_tracker(od_results, width, height);

                    if (!objs.empty()) {
                        g_active_tracks = g_tracker->update(objs);

                        for (const auto& track : g_active_tracks) {
                            if (track.tlwh.size() < 4) continue;
                            cv::Rect_<float> tr(track.tlwh[0], track.tlwh[1],
                                                track.tlwh[2], track.tlwh[3]);
                            float best_ov = 0; int best_lbl = -1;
                            for (const auto& o : objs) {
                                cv::Rect_<float> inter = tr & o.rect;
                                cv::Rect_<float> uni   = tr | o.rect;
                                if (uni.area() > 0) {
                                    float ov = inter.area() / uni.area();
                                    if (ov > best_ov) { best_ov = ov; best_lbl = o.label; }
                                }
                            }
                            if (best_ov > 0.3f && best_lbl >= 0)
                                g_track_labels[track.track_id] = best_lbl;
                        }
                        cleanup_stale_labels(g_active_tracks);
                    }
                }
            }

            uint64_t ai_time = get_time_us() - ai_start;

            // --------------------------------------------------------
            // Step 4: OSD (NV12 drawing)
            // --------------------------------------------------------
            uint64_t osd_start = get_time_us();

            for (const auto& track : g_active_tracks) {
                if (track.tlwh.size() < 4) continue;
                int sX = (int)track.tlwh[0], sY = (int)track.tlwh[1];
                int bw = (int)track.tlwh[2], bh = (int)track.tlwh[3];
                if (sX < 0 || sY < 0 || bw <= 10 || bh <= 10) continue;
                if (sX + bw > width || sY + bh > height) continue;

                draw_bbox_nv12(data, width, height, vir_width,
                               sX, sY, bw, bh, 2);

                char label[64];
                auto it = g_track_labels.find(track.track_id);
                if (it != g_track_labels.end() && it->second >= 0 && it->second < 80)
                    snprintf(label, sizeof(label), "ID:%d %s %.0f%%",
                             track.track_id, coco_cls_to_name(it->second),
                             track.score * 100);
                else
                    snprintf(label, sizeof(label), "ID:%d %.0f%%",
                             track.track_id, track.score * 100);

                draw_text_labels_hybrid(data, width, height, vir_width,
                                        sX, sY, label);
            }

            uint64_t osd_time = get_time_us() - osd_start;

            // --------------------------------------------------------
            // Step 5: Encode + stream
            // --------------------------------------------------------
            RK_MPI_SYS_MmzFlushCache(src_Blk, RK_FALSE);
            h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
            h264_frame.stVFrame.u64PTS     = TEST_COMM_GetNowUs();

            s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, 1000);
            if (s32Ret != RK_SUCCESS) {
                consecutive_errors++;
                if (consecutive_errors > 10) {
                    printf("ERROR: VENC consecutive failures\n");
                    g_running = false; break;
                }
                continue;
            }
            consecutive_errors = 0;

            // --------------------------------------------------------
            // Step 6: FPS / perf logging
            // --------------------------------------------------------
            frame_count++;
            fps_frame_count++;

            uint64_t now = get_time_us();
            if (now - last_fps_time >= 1000000) {
                double fps = (double)fps_frame_count /
                             ((now - last_fps_time) / 1000000.0);
                double elapsed = (now - start_time) / 1000000.0;

                if (run_inference) {
                    printf("[%6.1fs] FPS:%5.1f | AI:%5.1fms | "
                           "Pre:%5.1fms (%s) | OSD:%4.1fms | T:%lu\n",
                           elapsed, fps, ai_time / 1000.0,
                           preprocess_time / 1000.0,
                           used_rga ? "RGA" : "CPU",
                           osd_time / 1000.0,
                           (unsigned long)g_active_tracks.size());
                } else {
                    printf("[%6.1fs] FPS:%5.1f | AI: skip | "
                           "OSD:%4.1fms | T:%lu\n",
                           elapsed, fps, osd_time / 1000.0,
                           (unsigned long)g_active_tracks.size());
                }
                fps_frame_count = 0;
                last_fps_time = now;
            }
        }

        av_packet_unref(packet);
    }

    // ================================================================
    // Shutdown
    // ================================================================
    uint64_t end_time = get_time_us();
    double total = (end_time - start_time) / 1000000.0;
    double avg   = total > 0 ? (double)frame_count / total : 0;

    printf("\n========================================\n");
    printf("Summary: %lu frames, %.1fs, %.1f FPS avg\n",
           (unsigned long)frame_count, total, avg);
    printf("========================================\n");

    g_running = false;
    pthread_join(stream_thread, NULL);

    if (g_tracker) delete g_tracker;
    g_active_tracks.clear();
    g_track_labels.clear();

    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);

    if (bgr_frame) delete bgr_frame;
    if (temp_nv12_buffer) free(temp_nv12_buffer);

    if (packet)        av_packet_free(&packet);
    if (nv12_frame)    av_frame_free(&nv12_frame);
    if (avframe)       av_frame_free(&avframe);
    if (sws_ctx)       sws_freeContext(sws_ctx);
    if (codecContext)   avcodec_free_context(&codecContext);
    if (formatContext)  avformat_close_input(&formatContext);
    avformat_network_deinit();

    if (src_Blk  != MB_INVALID_HANDLE) RK_MPI_MB_ReleaseMB(src_Blk);
    if (src_Pool != MB_INVALID_POOLID) RK_MPI_MB_DestroyPool(src_Pool);

    if (g_rtsplive) rtsp_del_demo(g_rtsplive);
    RK_MPI_SYS_Exit();

    cleanup_rga_acceleration();
    release_yolov5_model(&rknn_app_ctx);
    deinit_post_process();

    printf("[OK] Cleanup complete\n");
    return 0;
}
