/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_PARSER_H_
#define LIGHTGBM_PARSER_H_

#include <vector>

namespace LightGBM {
/*! \brief Interface for Parser */
class Parser {
 public:
  /*! \brief virtual destructor */
  virtual ~Parser() {}

  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output columns, store in (column_idx, values)
  * \param out_label Label will store to this if exists
  * \param line_idx the line index of current line
  */
  virtual void ParseOneLine(const char* str,
                            std::vector<std::pair<int, double>>* out_features,
                            double* out_label, const int line_idx = -1) const = 0;

  virtual int NumFeatures() const = 0;

  /*!
  * \brief Create an object of parser, will auto choose the format depend on file
  * \param filename One Filename of data
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param label_idx index of label column
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, bool header, int num_features, int label_idx);
};

} // namespace LightGBM

#endif   // LightGBM_PARSER_H_