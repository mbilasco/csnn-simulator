#include "dataset/TwoStream.h"
using namespace dataset;

TwoStream::TwoStream(const std::string &folder_path, const size_t &method, const size_t &draw, size_t max_read) : _folder_path(folder_path), _method(method), _draw(draw),
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

    // load the saved spatial features.
    LoadPairVector(_data_list[0], _features_space);
    // load the saved temporal features.
    LoadPairVector(_data_list[1], _features_time);

    std::string _draw_fused_path = "";
    if (_draw == 1)
    {
        if (folder_path.find("/train") != std::string::npos)
            _draw_fused_path = folder_path.substr(0, folder_path.find("/train"));
        if (folder_path.find("/test") != std::string::npos)
            _draw_fused_path = folder_path.substr(0, folder_path.find("/test"));
        std::filesystem::create_directories(_draw_fused_path + "/F_R_" + std::to_string(_method) + "/space/");
        std::filesystem::create_directories(_draw_fused_path + "/F_R_" + std::to_string(_method) + "/time/");
        std::filesystem::create_directories(_draw_fused_path + "/F_R_" + std::to_string(_method) + "/concat/");
        _draw_fused_path = _draw_fused_path + "/F_R_" + std::to_string(_method) + "/";
    }

    // Fuse the temporal and spatial features together
    if (_method == 0)
        FuseStreamsConcat(features, _features_fused, _draw_fused_path);

    if (_method == 1)
        FuseStreamsConcat1(_features_space, _features_time, _features_fused, _draw_fused_path);
    if (_method == 2)
        FuseStreamsConcat2(_features_space, _features_time, _features_fused, _draw_fused_path);
    if (_method == 5)
        FuseStreamsConcat5(_features_space, _features_time, _features_fused, _draw_fused_path);
    if (_method == 6)
        FuseStreamsConcat6(_features_space, _features_time, _features_fused, _draw_fused_path);
    if (_method == 7)
        FuseStreamsConcat7(_features_space, _features_time, _features_fused, _draw_fused_path);
    if (_method == 3)
        FuseStreamsConcat3(_features_space, _features_time, _features_fused, _draw_fused_path);
    if (_method == 4)
        FuseStreamsConcat4(_features_space, _features_time, _features_fused, _draw_fused_path);

    _shape = _features_fused[0].second.shape();
    _size = _features_fused.size();
}

bool TwoStream::has_next() const
{
    return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> TwoStream::next()
{
    std::string _label = _features_fused[_cursor].first;

    std::pair<std::string, Tensor<InputType>> out(_label, _shape);

    out.second = _features_fused[_cursor].second;

    _cursor++;

    return out;
}

void TwoStream::reset()
{
    _cursor = 0;
}

void TwoStream::close()
{
}

size_t TwoStream::size() const
{
    return std::min(_size, _max_read);
}

std::string TwoStream::to_string() const
{
    return "TwoStream(" + _folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &TwoStream::shape() const
{
    return _shape;
}

uint32_t TwoStream::swap(uint32_t v)
{
    return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}
