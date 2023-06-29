#include "dataset/MultipleStream.h"
using namespace dataset;

MultipleStream::MultipleStream(const std::string &folder_path, const size_t &method, const size_t &draw, size_t max_read) : _folder_path(folder_path), _method(method), _draw(draw),
                                                                                                                  _size(0), _cursor(0), _shape({1, 1, 1, 1}), _max_read(max_read)
{
    // Get the saved locations of the featues
    for (const auto &file : std::filesystem::directory_iterator(_folder_path))
    {
        std::string _file_path = file.path();
        _data_list.push_back(_file_path);
    }

    size_t size = _data_list.size();
    std::vector<std::vector<std::pair<std::string, Tensor<float>>>> features;
    // load the saved features.
    for (int i = 0; i < size; i++)
        LoadPairVectors(_data_list[i], features);

    std::string _draw_fused_path = "";
    if (_draw == 1)
    {
        if (folder_path.find("/train") != std::string::npos)
            _draw_fused_path = folder_path.substr(0, folder_path.find("/train"));
        if (folder_path.find("/test") != std::string::npos)
            _draw_fused_path = folder_path.substr(0, folder_path.find("/test"));
        for (int i = 0; i < size; i++)
            std::filesystem::create_directories(_draw_fused_path + "/F_R_" + std::to_string(_method) + "/"+std::to_string(i)+"/");
        std::filesystem::create_directories(_draw_fused_path + "/F_R_" + std::to_string(_method) + "/concat/");
        _draw_fused_path = _draw_fused_path + "/F_R_" + std::to_string(_method) + "/";
    }

    // Fuse the temporal and spatial features together
    if (_method == 1)
    {
        FuseStreamsConcat(features, _features_fused, _draw_fused_path);
    }

    _shape = _features_fused[0].second.shape();
    _size = _features_fused.size();
}

bool MultipleStream::has_next() const
{
    return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> MultipleStream::next()
{
    std::string _label = _features_fused[_cursor].first;

    std::pair<std::string, Tensor<InputType>> out(_label, _shape);

    out.second = _features_fused[_cursor].second;

    _cursor++;

    return out;
}

void MultipleStream::reset()
{
    _cursor = 0;
}

void MultipleStream::close()
{
}

size_t MultipleStream::size() const
{
    return std::min(_size, _max_read);
}

std::string MultipleStream::to_string() const
{
    return "MultipleStream(" + _folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &MultipleStream::shape() const
{
    return _shape;
}

uint32_t MultipleStream::swap(uint32_t v)
{
    return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}
