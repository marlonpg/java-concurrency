package com.gamba.software.concurrency;

import java.util.concurrent.atomic.AtomicReference;

public class LockFreeStack<T> {
    private final AtomicReference<Node<T>> head = new AtomicReference<>();
    
    private static class Node<T> {
        final T data;
        final Node<T> next;
        
        Node(T data, Node<T> next) {
            this.data = data;
            this.next = next;
        }
    }
    
    public void push(T item) {
        Node<T> currentHead;
        Node<T> newNode;
        do {
            currentHead = head.get();
            newNode = new Node<>(item, currentHead);
        } while (!head.compareAndSet(currentHead, newNode));
    }
    
    public T pop() {
        Node<T> currentHead;
        Node<T> newHead;
        do {
            currentHead = head.get();
            if (currentHead == null) return null;
            newHead = currentHead.next;
        } while (!head.compareAndSet(currentHead, newHead));
        
        return currentHead.data;
    }
    
    public boolean isEmpty() {
        return head.get() == null;
    }
}