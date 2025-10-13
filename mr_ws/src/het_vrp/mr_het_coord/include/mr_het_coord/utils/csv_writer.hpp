#ifndef MR_HET_COORD_CVS_WRITER_HPP
#define MR_HET_COORD_CVS_WRITER_HPP

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Reference for variadic templates (typename... Args): https://en.cppreference.com/w/cpp/language/parameter_pack.html
class CSVWriter {
public:
  CSVWriter(std::string const& filename, std::vector<std::string> const& header) {
    bool exists = std::filesystem::exists(filename);
    file_.open(filename, exists ? std::ios::app : std::ios::out);
    if (!exists) {
      for (size_t i = 0; i < header.size(); ++i)
        file_ << (i ? "," : "") << header[i];
      file_ << "\n";
    }
  }

  template <typename... Args>
  void writeRow(Args const&... args) {
    writeFields(args...);
    file_ << "\n";
  }

private:
  std::ofstream file_;

  template <typename T>
  void writeFields(T const& value) { file_ << value; }

  template <typename T, typename... Args>
  void writeFields(T const& value, Args const&... rest) {
    file_ << value;
    if constexpr (sizeof...(rest) > 0) file_ << ",";
    writeFields(rest...);
  }
};

#endif  // MR_HET_COORD_CVS_WRITER_HPP