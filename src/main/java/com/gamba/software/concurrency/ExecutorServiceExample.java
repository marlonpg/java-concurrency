package com.gamba.software.concurrency;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.*;

public class ExecutorServiceExample {
    static Logger logger = LoggerFactory.getLogger(ExecutorServiceExample.class);
    private ExecutorService fixedPool = Executors.newFixedThreadPool(4);

    public static void main(String[] args) {
        // Fixed thread pool - reuses fixed number of threads
        ExecutorService fixedPool = Executors.newFixedThreadPool(4);

        // Cached thread pool - creates threads as needed
        ExecutorService cachedPool = Executors.newCachedThreadPool();

        // Single thread executor - guarantees sequential execution
        ExecutorService singleThread = Executors.newSingleThreadExecutor();

        // Submit tasks
        fixedPool.submit(() -> {
            logger.info("Task running in: " + Thread.currentThread().getName());
        });

        // Always shutdown executors
        fixedPool.shutdown();

        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);

        Runnable task = () -> logger.info("Running at " + System.currentTimeMillis());

        // Start after 2s, repeat every 5s (fixed rate)
        scheduler.scheduleAtFixedRate(task, 2, 5, TimeUnit.SECONDS);
    }
}
