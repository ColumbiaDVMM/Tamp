#ifndef _STRUCTURED_TYPED_DATA_
#define _STRUCTURED_TYPED_DATA_

#include <cstdint>
#include <vector>
#include <memory>

namespace structured {

typedef std::int64_t int64;

struct ExecutiveCore;

template <typename T> struct TypedData;

struct BufferedData {
  virtual ~BufferedData() = default;
  template<typename T> inline TypedData<T>& typed() {
    return dynamic_cast<TypedData<T>&>(*this);
  }
  template<typename T> inline const TypedData<T>& typed() const {
    return dynamic_cast<const TypedData<T>&>(*this);
  }
  virtual int64 size() const = 0;
  virtual int64 count() const = 0;
  virtual int dims() const = 0;
  virtual int64 dim_size(int index) const = 0;
  virtual void reshape(const std::vector<int64> shape) = 0;
  virtual std::shared_ptr<BufferedData>fromBuffer(void* buf) const = 0;
};

template <typename T>
struct TypedData: BufferedData {
  virtual ~TypedData() = default;
  virtual const T * data() const = 0;
  virtual T * data() = 0;
  virtual int64 size() const { return sizeof(T) * this->count(); }
};

}

#endif
