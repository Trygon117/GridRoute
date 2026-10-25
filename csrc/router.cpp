#include "router.h"

namespace gridroute {

Router::Router() {}

void Router::initialize(
    const std::vector<torch::Tensor>& storage_buffers,
    const std::vector<std::string>& slot_names,
    const std::vector<torch::Tensor>& slot_views
) {
    storage_buffers_ = storage_buffers;
    slot_views_.clear();
    for (size_t i = 0; i < slot_names.size(); ++i) {
        slot_views_[slot_names[i]] = slot_views[i];
    }
}

void Router::reset_memory() {
    for (auto& buf : storage_buffers_) {
        buf.zero_();
    }
}

} // namespace gridroute