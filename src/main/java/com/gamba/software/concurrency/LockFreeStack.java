package com.gamba.software.concurrency;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Lock-free stack implementation using atomic compare-and-swap operations.
 * Provides thread-safe stack operations without using locks or synchronization.
 */
public class LockFreeStack<T> {
    // Atomic reference to stack head - enables lock-free operations
    private final AtomicReference<Node<T>> head = new AtomicReference<>();
    
    /**
     * Immutable node structure for the stack.
     * Immutability is crucial for lock-free algorithms.
     */
    private static class Node<T> {
        final T data;      // Node's data payload
        final Node<T> next; // Reference to next node in stack
        
        Node(T data, Node<T> next) {
            this.data = data;
            this.next = next;
        }
    }
    
    /**
     * Pushes item onto stack using lock-free algorithm.
     * Retries until compare-and-swap succeeds.
     */
    public void push(T item) {
        Node<T> currentHead;
        Node<T> newNode;
        do {
            currentHead = head.get();  // Read current head
            newNode = new Node<>(item, currentHead); // Create new node pointing to current head
        } while (!head.compareAndSet(currentHead, newNode)); // Atomically update head if unchanged
    }
    
    /**
     * Pops item from stack using lock-free algorithm.
     * Returns null if stack is empty.
     */
    public T pop() {
        Node<T> currentHead;
        Node<T> newHead;
        do {
            currentHead = head.get(); // Read current head
            if (currentHead == null) return null; // Stack is empty
            newHead = currentHead.next; // Next node becomes new head
        } while (!head.compareAndSet(currentHead, newHead)); // Atomically update head
        
        return currentHead.data; // Return popped data
    }
    
    /**
     * Checks if stack is empty by examining head reference.
     */
    public boolean isEmpty() {
        return head.get() == null;
    }
}