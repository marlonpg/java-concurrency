package com.gamba.software.concurrency;

import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentCollectionExample {
    public static void main(String[] args) {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

        // Thread-safe operations
        map.put("key1", 1);
        map.putIfAbsent("key2", 2);

        // Atomic operations
        map.compute("key1", (key, value) -> value + 1);
        map.merge("key3", 1, Integer::sum);

        System.out.println(map);
    }
}
