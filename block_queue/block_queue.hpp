#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

template<typename T>
class BlockingQueue {
public:
    void push(T v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push(std::move(v));
        }
        cv_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop();
        return v;
    }

    template<typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout_duration) {
        std::unique_lock<std::mutex> lk(m_);
        bool success = cv_.wait_for(lk, timeout_duration, [&]{ return stop_ || !q_.empty(); });
        // 无论是超时还是停止，只要没数据就返回 nullopt
        if (!success && q_.empty()) return std::nullopt;
        if (stop_ && q_.empty()) return std::nullopt;
        
        T v = std::move(q_.front());
        q_.pop();
        return v;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    // 【新增】供外部查询是否已停止
    bool is_stopped() {
        std::lock_guard<std::mutex> lk(m_);
        return stop_;
    }

    bool empty() {
        std::lock_guard<std::mutex> lk(m_);
        return q_.empty();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<T> q_;
    bool stop_ = false;
};