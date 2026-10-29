/* Row-based PNG I/O into caller-owned mapped storage. No whole-image heap copy. */
#include <png.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int bg_png_read(const char *path, unsigned char *out, int width, int height, int channels) {
    FILE *file = fopen(path, "rb"); if (!file) return 1;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(file); return 2; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(file); return 3; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, NULL); fclose(file); return 4; }
    png_init_io(png, file); png_read_info(png, info);
    if (png_get_image_width(png,info)!=(png_uint_32)width || png_get_image_height(png,info)!=(png_uint_32)height) png_error(png,"size mismatch");
    int color = png_get_color_type(png,info), depth=png_get_bit_depth(png,info);
    if (depth == 16) png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    int alpha = (color & PNG_COLOR_MASK_ALPHA) || png_get_valid(png, info, PNG_INFO_tRNS);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (channels == 1) {
        /* Masks must already be grayscale: no undocumented color conversion. */
        if (color != PNG_COLOR_TYPE_GRAY && color != PNG_COLOR_TYPE_GRAY_ALPHA) png_error(png,"mask must be grayscale");
        if (alpha) png_set_strip_alpha(png);
    } else {
        if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
        if (channels == 3 && alpha) png_set_strip_alpha(png);
        if (channels == 4 && !alpha) png_set_add_alpha(png,255,PNG_FILLER_AFTER);
    }
    int passes=png_set_interlace_handling(png); png_read_update_info(png,info);
    if (png_get_rowbytes(png,info)!=(size_t)width*channels) png_error(png,"row layout mismatch");
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    for (int pass=0;pass<passes;pass++) {
        size_t released=0;
        for (int y=0;y<height;y++) {
            png_read_row(png,out+(size_t)y*width*channels,NULL);
            if ((y+1)%128==0) {
                size_t end=((size_t)(y+1)*width*channels/page)*page;
                if(end>released){msync(out+released,end-released,MS_SYNC);madvise(out+released,end-released,MADV_DONTNEED);released=end;}
            }
        }
    }
    png_read_end(png,NULL); png_destroy_read_struct(&png,&info,NULL); fclose(file); return 0;
}

int bg_png_write(const char *path, const unsigned char *data, int width, int height, int channels) {
    FILE *file=fopen(path,"wb"); if(!file) return 1;
    png_structp png=png_create_write_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
    if(!png){fclose(file);return 2;}
    png_infop info=png_create_info_struct(png);
    if(!info){png_destroy_write_struct(&png,NULL);fclose(file);return 3;}
    if(setjmp(png_jmpbuf(png))){png_destroy_write_struct(&png,&info);fclose(file);return 4;}
    int color=channels==1?PNG_COLOR_TYPE_GRAY:(channels==3?PNG_COLOR_TYPE_RGB:PNG_COLOR_TYPE_RGBA);
    png_init_io(png,file);png_set_IHDR(png,info,width,height,8,color,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png,1);png_write_info(png,info);
    for(int y=0;y<height;y++)png_write_row(png,(png_const_bytep)(data+(size_t)y*width*channels));
    png_write_end(png,NULL);png_destroy_write_struct(&png,&info);fclose(file);return 0;
}
