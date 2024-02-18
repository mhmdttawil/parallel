#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <float.h>
#include "timer.h"
#include "cmdline.h"

#define PROBLEM_SPACE 2
#define WINDOW_SIZE 800

#define MSG_WORK 1
#define MSG_DATA 2
#define MSG_STOP 3

typedef struct {
    double re, im;
} ComplexNum;

#include "mandelbrot-gui.h"

int run_master(int num_workers, int w, int h, double re_min, double re_max, double im_min, double im_max, int iterations);
int run_worker(int worker_id, int w, int h, double re_min, double re_max, double im_min, double im_max, int iterations);

int main(int argc, char *argv[]) {
    int num_procs, proc_id, exit_status, max_iters;
    double re_min = -PROBLEM_SPACE, re_max = PROBLEM_SPACE, im_min = -PROBLEM_SPACE, im_max = PROBLEM_SPACE;
    int w = WINDOW_SIZE, h = WINDOW_SIZE;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    if (num_procs < 2) {
        fprintf(stderr, "Need at least 2 processes.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char *help_msg = "Usage: %s maxiter [x0 y0 size]\n";
    max_iters = get_integer_arg_extended(argc, argv, 1, 1, "maxiter", help_msg, (proc_id == 0), (void (*)(void))MPI_Finalize);
    
    if (argc > 2) {
        double x0 = get_floating_arg_extended(argc, argv, 2, -DBL_MAX, "x0", help_msg, (proc_id == 0), (void (*)(void))MPI_Finalize);
        double y0 = get_floating_arg_extended(argc, argv, 3, -DBL_MAX, "y0", help_msg, (proc_id == 0), (void (*)(void))MPI_Finalize);
        double size = get_floating_arg_extended(argc, argv, 4, 0, "size", help_msg, (proc_id == 0), (void (*)(void))MPI_Finalize);
        re_min = x0 - size;
        re_max = x0 + size;
        im_min = y0 - size;
        im_max = y0 + size;
    }

    if (proc_id == 0) {
        exit_status = run_master(num_procs - 1, w, h, re_min, re_max, im_min, im_max, max_iters);
    } else {
        exit_status = run_worker(proc_id, w, h, re_min, re_max, im_min, im_max, max_iters);
    }

    MPI_Finalize();
    return exit_status;
}

int run_master(int workers, int w, int h, double re_min, double re_max, double im_min, double im_max, int iters) {
    Display *disp;
    Window window;
    GC gc;
    long color_min = 0, color_max = 0;
    int row_current, row_next;
    double time_start, time_end;
    long *message_data = (long *)malloc((w + 1) * sizeof(*message_data));
    MPI_Status stat;
    int tasks_remaining, worker_id;
    int gui_setup;

    gui_setup = setup(w, h, &disp, &window, &gc, &color_min, &color_max);
    time_start = get_time();

    MPI_Bcast(&color_min, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color_max, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    row_next = 0;
    tasks_remaining = 0;

    for (int i = 0; i < workers; ++i) {
        MPI_Send(&row_next, 1, MPI_INT, i + 1, MSG_WORK, MPI_COMM_WORLD);
        ++row_next;
        ++tasks_remaining;
    }

    while (tasks_remaining > 0) {
        MPI_Recv(message_data, w + 1, MPI_LONG, MPI_ANY_SOURCE, MSG_DATA, MPI_COMM_WORLD, &stat);

        --tasks_remaining;
        worker_id = stat.MPI_SOURCE;

        if (row_next < h) {
            MPI_Send(&row_next, 1, MPI_INT, worker_id, MSG_WORK, MPI_COMM_WORLD);
            ++row_next;
            ++tasks_remaining;
        } else {
            MPI_Send(&row_next, 0, MPI_INT, worker_id, MSG_STOP, MPI_COMM_WORLD);
        }

        row_current = message_data[0];
        for (int col = 0; col < w; ++col) {
            if (gui_setup == EXIT_SUCCESS) {
                XSetForeground(disp, gc, message_data[col + 1]);
                XDrawPoint(disp, window, gc, col, row_current);
            }
        }
    }

    if (gui_setup == EXIT_SUCCESS) {
        XFlush(disp);
    }

    time_end = get_time();
    printf("\nDynamic task distribution MPI program\n");
    printf("Worker process count = %d\n", workers);
    printf("Center = (%g, %g), Size = %g\n", (re_max + re_min) / 2, (im_max + im_min) / 2, (re_max - re_min) / 2);
    printf("Max iterations = %d\n", iters);
    printf("Execution time = %g seconds\n", time_end - time_start);
    printf("\n");

    if (gui_setup == EXIT_SUCCESS) {
        interact(disp, &window, w, h, re_min, re_max, im_min, im_max);
    }

    free(message_data);
    return EXIT_SUCCESS;
}

int run_worker(int worker_id, int w, int h, double re_min, double re_max, double im_min, double im_max, int iters) {
    MPI_Status stat;
    int row;
    long color_min, color_max;
    double scale_re, scale_im, scale_color;
    long *message_data = (long *)malloc((w + 1) * sizeof(*message_data));

    MPI_Bcast(&color_min, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&color_max, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    scale_re = (re_max - re_min) / w;
    scale_im = (im_max - im_min) / h;
    scale_color = (double)(color_max - color_min) / (iters - 1);

    while (true) {
        MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        if (stat.MPI_TAG == MSG_STOP) {
            break;
        }
        message_data[0] = row;
        for (int col = 0; col < w; ++col) {
            ComplexNum c, z = {0.0, 0.0};
            double temp;
            int k;
            c.re = re_min + col * scale_re;
            c.im = im_min + (h - row - 1) * scale_im;
            k = 0;
            while (k < iters) {
                temp = z.re * z.re - z.im * z.im + c.re;
                z.im = 2.0 * z.re * z.im + c.im;
                z.re = temp;
                if ((z.re * z.re + z.im * z.im) > 4.0) break;
                k++;
            }
            long color = (k == iters) ? color_max : (long)((double)k / iters * (color_max - color_min)) + color_min;
            message_data[col + 1] = color;
        }
        MPI_Send(message_data, w + 1, MPI_LONG, 0, MSG_DATA, MPI_COMM_WORLD);
    }

    free(message_data);
    return EXIT_SUCCESS;
}
