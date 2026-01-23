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

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

// RTSP input URL
#define RTSP_INPUT_URL "rtsp://220.254.72.200/Src/MediaInput/h264/stream_2"

// Get current time in microseconds
uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char *argv[]) {
    // FFmpeg variables
    AVFormatContext *formatContext = NULL;
    AVCodecContext *codecContext = NULL;
    AVCodec *codec = NULL;
    AVPacket packet;
    AVFrame *avframe = NULL;
    int videoStreamIndex = -1;

    // Stream dimensions
    int width = 0;
    int height = 0;

    // FPS tracking
    uint64_t frame_count = 0;
    uint64_t start_time = 0;
    uint64_t last_fps_time = 0;
    uint64_t fps_frame_count = 0;

    // Initialize FFmpeg network
    avformat_network_init();

    // Open RTSP stream
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);

    printf("========================================\n");
    printf("RTSP Stream Processing Monitor\n");
    printf("========================================\n");
    printf("Connecting to: %s\n", RTSP_INPUT_URL);
    
    if (avformat_open_input(&formatContext, RTSP_INPUT_URL, NULL, &opts) != 0) {
        fprintf(stderr, "ERROR: Failed to open RTSP stream\n");
        av_dict_free(&opts);
        return 1;
    }
    av_dict_free(&opts);
    printf("✓ Connected successfully\n");

    // Find stream information
    printf("Finding stream information...\n");
    if (avformat_find_stream_info(formatContext, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to find stream information\n");
        avformat_close_input(&formatContext);
        avformat_network_deinit();
        return 1;
    }
    printf("✓ Stream info found\n");

    // Find the first video stream
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            width = formatContext->streams[i]->codecpar->width;
            height = formatContext->streams[i]->codecpar->height;
            printf("✓ Video stream found:\n");
            printf("  - Resolution: %dx%d\n", width, height);
            printf("  - Codec: %s\n", avcodec_get_name(formatContext->streams[i]->codecpar->codec_id));
            break;
        }
    }

    if (videoStreamIndex == -1) {
        fprintf(stderr, "ERROR: No video stream found\n");
        avformat_close_input(&formatContext);
        avformat_network_deinit();
        return 1;
    }

    // Setup decoder
    codecContext = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar);

    codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec) {
        fprintf(stderr, "ERROR: Codec not found\n");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_network_deinit();
        return 1;
    }

    if (avcodec_open2(codecContext, codec, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to open codec\n");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        avformat_network_deinit();
        return 1;
    }
    printf("✓ Decoder initialized\n");

    avframe = av_frame_alloc();
    av_init_packet(&packet);

    printf("\n========================================\n");
    printf("Starting frame processing...\n");
    printf("Press Ctrl+C to stop\n");
    printf("========================================\n\n");

    start_time = get_time_us();
    last_fps_time = start_time;

    // Main loop - read from RTSP stream
    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            // Decode video frame
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                while (avcodec_receive_frame(codecContext, avframe) == 0) {
                    frame_count++;
                    fps_frame_count++;

                    uint64_t current_time = get_time_us();
                    uint64_t elapsed_us = current_time - last_fps_time;

                    // Calculate and print FPS every second
                    if (elapsed_us >= 1000000) {  // 1 second
                        double fps = (double)fps_frame_count / (elapsed_us / 1000000.0);
                        double total_elapsed = (current_time - start_time) / 1000000.0;
                        
                        printf("[%.1fs] Frame: %lu | FPS: %.2f | Resolution: %dx%d\n",
                               total_elapsed,
                               (unsigned long)frame_count,
                               fps,
                               avframe->width,
                               avframe->height);
                        
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
    double avg_fps = (double)frame_count / total_time;

    printf("\n========================================\n");
    printf("Stream ended\n");
    printf("========================================\n");
    printf("Statistics:\n");
    printf("  - Total frames: %lu\n", (unsigned long)frame_count);
    printf("  - Total time: %.2f seconds\n", total_time);
    printf("  - Average FPS: %.2f\n", avg_fps);
    printf("========================================\n");

    // Cleanup
    av_frame_free(&avframe);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    avformat_network_deinit();

    return 0;
}