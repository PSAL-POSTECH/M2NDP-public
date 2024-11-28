#ifndef LATENCTY_QUEUE
#define LATENCTY_QUEUE

#include <deque>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <string>
#include <unordered_map>

enum latency_queue_stat {
    ACTIVE,
    IDLE,
    STALLED
};

template<typename T>
class latency_queue {
    public:
        latency_queue(
            uint32_t head_latency = 1, uint32_t flit_size = 32, std::string name = "",
            uint32_t clock_diver = 1, uint32_t size_limit = UINT32_MAX);
        bool push_back(T* mem_fetch, uint64_t cur_clock);
        void pop_front();
        T* back();
        T* front();
        bool is_arrived(uint64_t cur_clock);
        bool is_idle();
        size_t size();
        void increase_stat(latency_queue_stat key);
        uint64_t get_stat(latency_queue_stat key);
        uint64_t total_stat();
        void clear_stat();
        void print_stat();

        int print_stat_freq;
    private:
        uint32_t head_latency;
        uint32_t flit_size;
        uint32_t clock_diver;
        uint32_t size_limit;
        std::string name;
        std::deque<std::pair<T*, uint64_t>> queue;
        std::unordered_map<latency_queue_stat, uint64_t> stat_counter;
};

template<typename T>
latency_queue<T>::latency_queue(
    uint32_t head_latency, uint32_t flit_size, std::string name,
    uint32_t clock_diver, uint32_t size_limit)
    : head_latency(head_latency), flit_size(flit_size), name(name),
      clock_diver(clock_diver), size_limit(size_limit) {
    stat_counter[ACTIVE] = 0;
    stat_counter[IDLE] = 0;
    stat_counter[STALLED] = 0;
    print_stat_freq = 0;
    assert(flit_size);
}

template<typename T>
bool latency_queue<T>::push_back(T* mem_fetch, uint64_t cur_clock) {
    uint64_t latency;

    /* Check lenght of queue */
    if (size_limit <= size())
        return false;

    /* Note: Template variable T has the size() function! */
    latency = (mem_fetch->get_size() + flit_size - 1) / flit_size;
    if (!size()) {
        latency += (head_latency + cur_clock) * clock_diver;
        queue.push_back(std::pair<T*, uint64_t>(mem_fetch, latency));
    } else {
        latency += std::max(queue.back().second, (head_latency + cur_clock) * clock_diver);
        queue.push_back(std::pair<T*, uint64_t>(mem_fetch, latency));
    }
    return true;
}

template<typename T>
void latency_queue<T>::pop_front(){
    return queue.pop_front();
}

template<typename T>
T* latency_queue<T>::back() {
    return queue.back().first;
}

template<typename T>
T* latency_queue<T>::front() {
    return queue.front().first;
}

template<typename T>
bool latency_queue<T>::is_arrived(uint64_t cur_clock) {
    if (queue.size() && queue.front().second <= cur_clock * clock_diver)
        return true;
    return false;
}

template<typename T>
size_t latency_queue<T>::size() {
    return queue.size();
}

template<typename T>
bool latency_queue<T>::is_idle() {
    if (!size())
        return true;
    return false;
}

template<typename T>
void latency_queue<T>::increase_stat(latency_queue_stat key) {
    stat_counter[key]++;
    print_stat();
}

template<typename T>
uint64_t latency_queue<T>::get_stat(latency_queue_stat key) {
    return stat_counter.at(key);
}

template<typename T>
uint64_t latency_queue<T>::total_stat() {
    return stat_counter[ACTIVE] + stat_counter[IDLE] + stat_counter[STALLED];
}

template<typename T>
void latency_queue<T>::clear_stat() {
    stat_counter[ACTIVE] = 0;
    stat_counter[IDLE] = 0;
    stat_counter[STALLED] = 0;
}

template<typename T>
void latency_queue<T>::print_stat() {
    if (print_stat_freq && print_stat_freq == total_stat()) {
        int active_stat = stat_counter[ACTIVE];
        int idle_stat = stat_counter[IDLE];
        int stalled_stat = stat_counter[STALLED];
        int total_nr_stat = total_stat();

        std::cout << "[" << name << "] Utilization " <<
            active_stat * 100.0f / total_nr_stat <<
            "% IDLE " <<
            idle_stat * 100.0f / total_nr_stat <<
            "% queue size " << size() << std::endl;
        clear_stat();
    }
}

#endif