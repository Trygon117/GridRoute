#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>

namespace gridroute {

class Router {
public:
    Router();
    ~Router() = default;

    void initialize(
        const std::vector<torch::Tensor>& storage_buffers,
        const std::vector<std::string>& slot_names,
        const std::vector<torch::Tensor>& slot_views
    );

    void reset_memory();

private:
    std::vector<torch::Tensor> storage_buffers_;
    std::unordered_map<std::string, torch::Tensor> slot_views_;
};

} // namespace gridroute