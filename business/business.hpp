#pragma once

#include <iostream>
#include <queue>
#include <map>
#include <mutex>
#include <thread>
#include <functional>
#include <atomic>
#include <memory>
#include "block_queue.hpp" // 确保引用了你的阻塞队列头文件

// 模板类 T 通常是 InferOut
template <typename T>
class OrderedProcessor {
public:
    // 回调函数定义: (frame_id, 数据对象)
    using ProcessCallback = std::function<void(uint64_t, T&&)>;

    OrderedProcessor(BlockingQueue<T>& input_queue, 
                     size_t max_cache_size, 
                     size_t timeout_tolerance,
                     ProcessCallback callback)
        : input_queue_(input_queue), 
          max_cache_size_(max_cache_size), 
          timeout_tolerance_(timeout_tolerance), // 这里暂未通过时间判断，而是通过缓存大小判断跳帧
          callback_(callback) 
    {
    }

    ~OrderedProcessor() {
        Stop();
    }

    // =========================================================================
    // 【关键修复】这就是你报错缺失的函数
    // =========================================================================
    // 用于接收外部传入的共享原子变量指针
    void set_expect_id_notifier(std::shared_ptr<std::atomic<uint64_t>> ptr) {
        expect_id_ptr_ = ptr;
        // 初始化一下，告诉上游我现在想要第0帧
        if (expect_id_ptr_) {
            expect_id_ptr_->store(next_expect_id_, std::memory_order_relaxed);
        }
    }
    // =========================================================================

    void Start() {
        if (running_) return;
        running_ = true;
        worker_thread_ = std::thread(&OrderedProcessor::RunLoop, this);
    }

    void Stop() {
        running_ = false;
        input_queue_.stop(); // 让 input_queue 的 pop 返回 false
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    void Join() {
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

private:
    void RunLoop() {
        while (running_) {
            // 1. 从中间队列取数据
            auto item_opt = input_queue_.pop();
            if (!item_opt) break; // 队列停止

            T item = std::move(*item_opt);
            uint64_t fid = item.frame_id;

            // 2. 处理逻辑
            if (fid == next_expect_id_) {
                // A. 刚好是我要的那一帧 -> 直接输出
                callback_(fid, std::move(item));
                next_expect_id_++;
                
                // 检查缓存里有没有后续的帧 (比如输出了100，看看缓存里有没有101)
                process_stash();
            } 
            else if (fid > next_expect_id_) {
                // B. 未来的帧 (乱序到达) -> 存入缓存
                stash_[fid] = std::move(item);
                
                // 【防死锁机制 / 追帧逻辑】
                // 如果缓存太大，说明缺帧了（比如等100帧，结果来了105,106...125，100还没来）
                // 此时必须放弃等待，直接跳到缓存里最小的那一帧
                if (stash_.size() > max_cache_size_) {
                    // 找到缓存里最小的一帧
                    auto it = stash_.begin();
                    uint64_t jump_to_id = it->first;
                    
                    // 警告日志 (可选)
                    // std::cerr << "[Warning] Skip frames from " << next_expect_id_ << " to " << jump_to_id << std::endl;

                    // 强制更新期望ID
                    next_expect_id_ = jump_to_id;
                    
                    // 重新触发处理逻辑
                    process_stash();
                }
            } 
            else {
                // C. 过期的帧 (fid < next_expect_id_) -> 直接丢弃
                // 这种情况通常被 NPU 里的反压机制拦截了，但如果漏网之鱼到了这里，直接忽略
                // std::cout << "Drop old frame in processor: " << fid << std::endl;
            }

            // =================================================================
            // 【关键逻辑】实时同步反压信号
            // 每处理完一次数据，无论是否输出，都更新一下“我现在到底想要几号帧”
            // NPU 线程看到这个值变大后，就会自动丢弃比它小的老帧
            // =================================================================
            if (expect_id_ptr_) {
                expect_id_ptr_->store(next_expect_id_, std::memory_order_relaxed);
            }
        }
    }

    // 处理缓存中的连续帧
    void process_stash() {
        while (true) {
            auto it = stash_.find(next_expect_id_);
            if (it == stash_.end()) {
                break; // 缓存里没有下一帧，断了，继续等新数据
            }

            // 有下一帧 -> 输出
            callback_(it->first, std::move(it->second));
            stash_.erase(it);
            next_expect_id_++;
        }
    }

private:
    BlockingQueue<T>& input_queue_;
    size_t max_cache_size_;
    size_t timeout_tolerance_;
    ProcessCallback callback_;

    std::atomic<bool> running_{false};
    std::thread worker_thread_;

    // 内部状态
    uint64_t next_expect_id_ = 0;
    std::map<uint64_t, T> stash_; // 使用 map 自动按 frame_id 排序

    // 共享的反压信号指针
    std::shared_ptr<std::atomic<uint64_t>> expect_id_ptr_ = nullptr;
};