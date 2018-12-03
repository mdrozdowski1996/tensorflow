#ifndef TENSORFLOW_CONTRIB_TENSORRT_LRU_CACHE_H_
#define TENSORFLOW_CONTRIB_TENSORRT_LRU_CACHE_H

#include<deque>
#include<unordered_map>

#include "tensorflow/core/lib/core/errors.h"


namespace tensorflow {
namespace tensorrt {

template <class Key, class Value > class LRUCache{
public:
	typedef Value value_type;
	typedef Key key_type;
	
public:
	typedef typename std::unordered_map<key_type, value_type>::iterator map_iterator;
	typedef typename std::deque<key_type>::iterator deque_iterator;


	LRUCache() : capacity_(0) {}
	explicit LRUCache(size_t capacity) : capacity_(capacity) {}

	size_t capacity() const { return capacity_; }

	void reserve(size_t capacity) {
		capacity_ = capacity;
		DiscardOld();
	}

	size_t size() const { return objects_.size(); }

	size_t count(const key_type& k) const { return objects_.count(k); }
	
	value_type & at(key_type k, tensorflow::Status* status_ptr = nullptr) {
		tensorflow::Status status = Touch(k);
		if (!status.ok()) {
			if (status_ptr) {
				*status_ptr = status;
			}
			return not_found_value_;
		}
		return objects_.at(k);
	}

	map_iterator begin() {
		return objects_.begin();
	}

	map_iterator end() {
		return objects_.end();
	}

	template <typename... Args>
	std::pair<map_iterator, bool> emplace(Args&&... args) {
		DiscardOld(1);
		std::pair<map_iterator, bool> result =
			objects_.emplace(std::forward<Args>(args)...);
		key_type key = result.first->first;

		if (result.second){
			keys_.push_front(key);
		}
		else{
			Touch(key);
		}

		return result;
	}

private:
	std::unordered_map<key_type, value_type> objects_;
	std::deque<key_type> keys_;
	size_t capacity_;
	value_type not_found_value_;

	tensorflow::Status Touch(const key_type& k) {
		
		if (!count(k)) {
			return tensorflow::errors::NotFound("Key not found in cache");
		}
        
        	deque_iterator rank = std::find(keys_.begin(), keys_.end(), k);
		
        	if (rank != keys_.begin()) {
			keys_.erase(rank);
			keys_.push_front(k);
		}
		
		return tensorflow::Status::OK();
	}

	//creating n free positions in cache
	void DiscardOld(size_t n = 0) {
		if (n > capacity_) {
			LOG(ERROR) << "Insufficient capacity in cache";
			return;
		}
		while (objects_.size() > capacity_ - n) {
			key_type discard_key = keys_.back();
			keys_.pop_back();
			objects_.erase(discard_key);
		}
	}
};
}
}

#endif
