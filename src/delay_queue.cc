#ifdef TIMING_SIMULATION
#include "delay_queue.h"

#include "common.h"
#include "mem_fetch.h"
namespace NDPSim {

template <typename T>
void DelayQueue<T>::push(T data, int delay) {
  assert(m_only_latency);
  m_size++;
  m_queue.push(QueueEntry{data, m_cycle + delay});
}

template <typename T>
void DelayQueue<T>::push(T data, int delay, int interval) {
  assert(m_issued == false);
  m_size++;
  m_queue.push(QueueEntry{data, m_cycle + delay});
  if(!m_only_latency) m_issued = true;
  m_interval = interval;
}

template <typename T>
void DelayQueue<T>::pop() {
  assert(!empty());
  m_queue.pop();
  m_size--;
}

template <typename T>
T DelayQueue<T>::top() {
  assert(!empty());
  return m_queue.front().data;
}

template <typename T>
bool DelayQueue<T>::empty() {
  return m_queue.empty() || (m_queue.front().finish_cycle > m_cycle);
}

template <typename T>
bool DelayQueue<T>::queue_empty() {
  return m_queue.empty();
}

template <typename T>
bool DelayQueue<T>::full() {
  return m_issued || (m_max_size > 0 && m_size >= m_max_size);
}

template <typename T>
void DelayQueue<T>::cycle() {
  if (m_interval > 0) m_interval--;
  if (m_interval <= 0) m_issued = false;
  m_cycle++;
}

template class DelayQueue<std::pair<NdpInstruction, Context>>;
template class DelayQueue<RequestInfo*>;
template class DelayQueue<NDPSim::mem_fetch*>;
}  // namespace NDPSim

#endif