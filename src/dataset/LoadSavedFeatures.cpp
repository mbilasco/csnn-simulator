#include "dataset/LoadSavedFeatures.h"
using namespace dataset;

LoadSavedFeatures::LoadSavedFeatures(const std::string &folder_path, const size_t &draw, size_t max_read) : _folder_path(folder_path), _draw(draw), _vis_path(""),
                                                                                                            _size(0), _cursor(0), _shape({1, 1, 1, 1}), _max_read(max_read)
{
    // Get the saved locations of the featues
    for (const auto &file : std::filesystem::directory_iterator(_folder_path))
    {
        std::string _file_path = file.path();
        _data_list.push_back(_file_path);
    }

    // load the saved spatial features.
    LoadPairVector(_data_list[0], _features);

    if (_draw == 1)
    {
        std::size_t pos = _folder_path.size();
        _vis_path = _folder_path.substr(0, _folder_path.find_last_of('/', pos - 3));
        _vis_path2 = _folder_path.substr(_folder_path.find_last_of('/', pos - 3));
        std::filesystem::create_directories(_vis_path + "/Visualization/" + _vis_path2);
    }
    _shape = _features[0].second.shape();
    _size = _features.size();
}

bool LoadSavedFeatures::has_next() const
{
    return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> LoadSavedFeatures::next()
{
    std::string _label = _features[_cursor].first;

    std::pair<std::string, Tensor<InputType>> out(_label, _shape);

    out.second = _features[_cursor].second;

    _cursor++;

    if (_draw == 1 )//&& _label == "0" ) //  && _cursor < 100)
    {
        // Tensor<float>::normalize_tensor_by_max(out.second);
        // Tensor<float>::normalize_tensor_log(out.second);
        Tensor<float>::draw_one_k1_tensor(_vis_path + "/Visualization/" + _vis_path2 + _label + "_sample:" + std::to_string(_cursor), out.second);
    }
    return out;
}

void LoadSavedFeatures::reset()
{
    _cursor = 0;
}

void LoadSavedFeatures::close()
{
}

size_t LoadSavedFeatures::size() const
{
    return std::min(_size, _max_read);
}

std::string LoadSavedFeatures::to_string() const
{
    return "LoadSavedFeatures(" + _folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &LoadSavedFeatures::shape() const
{
    return _shape;
}

uint32_t LoadSavedFeatures::swap(uint32_t v)
{
    return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}
