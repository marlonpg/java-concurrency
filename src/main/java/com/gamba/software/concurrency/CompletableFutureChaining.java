package com.gamba.software.concurrency;

import java.util.concurrent.CompletableFuture;

public class CompletableFutureChaining {
    public static void main(String[] args) throws InterruptedException {
        CompletableFuture<String> result = CompletableFuture
                .supplyAsync(() -> {
                    try { Thread.sleep(100); } catch (Exception e) {}
                    return "Hello";
                })
                .thenApply(s -> s + " World")
                .thenApply(String::toUpperCase)
                .thenCompose(s -> CompletableFuture.supplyAsync(() -> s + "!"))
                .exceptionally(throwable -> "Error: " + throwable.getMessage());

        result.thenAccept(System.out::println);

        System.out.println("Main thread is free");
        Thread.sleep(200);
    }
}