// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#if __has_include("getopt.h")
#  include <getopt.h>
#else
#  error
#endif
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
namespace util
{
  namespace arg_type
  {
    static constexpr int none     = 0; /** enable/disable things, e.g. help */
    static constexpr int required = 1; /** argument for setting variables */
    static constexpr int optional = 2; /** unclear */
  }                                    // namespace arg_type
  using option_map_ = std::map<char, std::any>;
  using option_map  = std::shared_ptr<option_map_>;

  /** Initialize a default config map with help entry and return it.
   */
  inline option_map
  get_option_map();

  struct option_acc
  {
    /** Simple wrapper for getting the value of a config map key.
     */
    template <typename T>
    T
    at(char key) const;
    template <typename T>
    T
    unsafe_at(char key) const;

    option_map opts = get_option_map();
  };

  inline option_acc default_option_map;

  template <typename T, typename = void>
  /** Metafunction checking if a type is iterable.
   * If we can call std::begin() and std::end() on the type it is iterable.
   */
  struct is_iterable : std::false_type
  {};
  template <typename T>
  struct is_iterable<T,
                     std::void_t<decltype(std::begin(std::declval<T>())),
                                 decltype(std::end(std::declval<T>()))>>
    : std::true_type
  {};

  template <typename T, typename = void>
  /** Metafunction checking if a type is optional.
   */
  struct is_optional : std::false_type
  {};
  template <typename T>
  struct is_optional<T,
                     std::void_t<decltype(std::declval<T>().value_or(nullptr)),
                                 decltype(std::declval<T>().value()),
                                 decltype(std::declval<T>().has_value()),
                                 typename T::value_type>> : std::true_type
  {};

  template <class T>
  struct is_vector : std::false_type
  {};
  template <class T>
  struct is_vector<std::vector<T>> : std::true_type
  {};

  template <typename T>
  /** Function for extracting var from string optarg.
   */
  bool
  set(T &var, const std::string &optarg);

  /** Simple wrapper for GNU getopt.
   * Preferred use is in a structured block and letting the destructor
   * do the parsing. Example:
   *
   *     size_t              v1;
   *     int                 v2;
   *     double              v3;
   *     std::vector<double> v4(2, 0);
   *     util::option_acc    optacc;
   *     {
   *       util::cl_options clo(argc, argv, optacc.opts);
   *       clo.insert(v1, "aa", util::arg_type::required, 'a', "a size_t");
   *       clo.insert(v2, "bb", util::arg_type::required, 'b', "an int");
   *       clo.insert(v3, "cc", util::arg_type::required, 'c', "a double");
   *       clo.insert(v4, "dd", util::arg_type::required, 'd', "a vector");
   *     }
   *
   * The constructor takes argc and argv, then the options are inserted
   * and upon deconstruction they are populated.
   * Note that if you want to set iterable types (e.g. vectors or arrays)
   * you have to escape the spaces. Above example could be called like the
   * following:
   *
   *     ./a.out --aa=1 -d '1 2' --bb=1 -c 2.5
   *
   * Furthermore, cl_options will only allocate memory for std::vector.
   * For other iterables make sure enough space is available beforehand.
   * Iterables refer to objects on which std::begin() and std::end() can be
   * called. This is true for strings, but the set method has an extra overload
   * for a simlpe copy. Furthermore there are iterable types e.g. std::map,
   * which cannot use the current implementation. A new overload of the set
   * method would be able to fix this, if needed. Thus std::vector and
   * std::array or other sequence containers should be preferred.
   */
  struct cl_options
  {
    cl_options() = default;

    /** Constructor initializing argc and argv members.
     */
    cl_options(int argc, char **argv, option_map cm = nullptr)
      : argc_(argc)
      , argv_(argv)
      , config(cm)
    {
      insert(help_on_destruct,
             "help",
             arg_type::none,
             'h',
             "Show this help message");
    };

    /** Destructor which parses commandline args if not default initialized.
     */
    inline ~cl_options();

    template <typename T>
    /** Inserts new option which could appear in commandline args
     * @param var Variable to be set by the option
     * @param long_opt Long version of the option i.e. --long_opt=var
     * @param int ha argument type: none required optional in namespace arg_type
     * @param short_opt Short version of the option i.e. -short_opt var
     * @param help_msg Help message explaining the option; will be appended to the
     * help
     */
    void
    insert(T                 &var,
           const std::string &long_opt,
           int                ha,
           char               short_opt,
           const std::string &help_msg = "");

    /** Operator parsing commandline args and setting variables
     * @param argc argument count
     * @param argv argument values
     */
    inline bool
    operator()(int argc, char **argv);

    /** Returns help string
     */
    inline std::string
    get_help() const
    {
      return help;
    }

    /** Checks if the given short op is already in use.
     */
    bool
    exists(char short_opt) const
    {
      return events.find(short_opt) != events.end();
    }

  private:
    struct opt_info
    {
      std::string                              long_name;
      bool                                     matched;
      std::function<bool(const std::string &)> parse;
    };
    std::map<char, opt_info>       events;
    std::vector<struct option>     options;
    std::string                    searchstring;
    std::string                    help{"Available options:\n"};
    bool                           help_on_destruct{false};
    const int                      argc_{0};
    char                         **argv_;
    static constexpr struct option null_opt
    {
      nullptr, 0, nullptr, 0
    };
    option_map config;
  };

  cl_options::~cl_options()
  {
    if (argc_ && (!operator()(argc_, argv_) || help_on_destruct))
      {
        std::cerr << get_help();
        std::exit(EXIT_SUCCESS);
      }
    for (auto &el : options)
      {
        delete[] el.name;
        delete[] el.flag;
      }
  }

  template <typename T>
  void
  cl_options::insert(T                 &var,
                     const std::string &long_opt,
                     int                ha,
                     char               short_opt,
                     const std::string &help_msg)
  {
    if (!short_opt && long_opt.empty())
      {
        std::cerr << "option is zero or empty\n";
        return;
      }
    if (events.find(short_opt) != events.end())
      {
        std::cerr << "option -" << short_opt
                  << " already registered - skipping\n";
        return;
      }
    else
      {
        char *lo = new char[long_opt.size() + 1];
        std::copy(long_opt.begin(), long_opt.end(), lo);
        lo[long_opt.size()] = '\0';
        const struct option opt
        {
          lo, ha, NULL, short_opt
        };
        options.emplace_back(opt);
        searchstring += short_opt;
        if (config)
          (*config)[short_opt] = var;
        if (ha)
          {
            searchstring += ':';
            events.emplace(short_opt,
                           opt_info{long_opt,
                                    false,
                                    [&var, short_opt, this](
                                      const std::string &optarg) {
                                      bool success = set<T>(var, optarg);
                                      if (config)
                                        (*config)[short_opt] = var;
                                      return success;
                                    }});
          }
        else
          {
            if constexpr (std::is_same<bool, T>::value)
              events.emplace(short_opt,
                             opt_info{long_opt,
                                      false,
                                      [&var, short_opt, this](
                                        const std::string &) {
                                        var = true;
                                        if (config)
                                          (*config)[short_opt] = var;
                                        return true;
                                      }});
            else
              std::cerr << "option -" << short_opt
                        << " is not a bool but it should be.\n";
          }
        char *tmp = new char[help_msg.size() + 30];
        std::sprintf(tmp,
                     " -%-5c--%-19s%s\n",
                     short_opt,
                     long_opt.c_str(),
                     help_msg.c_str());
        help += tmp;
        delete[] tmp;
      }
  }

  bool
  cl_options::operator()(int argc, char **argv)
  {
    // reset global state of getopt ( needed in case of repeated parsing )
    optind = 1;
    options.emplace_back(null_opt);
    while (1)
      {
        int option_index = 0;
        int c            = getopt_long(
          argc, argv, searchstring.c_str(), options.data(), &option_index);
        if (-1 == c)
          break;
        if (auto event = events.find(static_cast<char>(c));
            event != events.end())
          {
            using namespace std::string_literals;
            opt_info         &info     = event->second;
            const std::string opt_name = info.long_name.empty() ?
                                           "-"s + event->first :
                                           ("--" + info.long_name);

            if (info.matched)
              std::cerr << "[Warning] Option " << opt_name
                        << " appears more than once. Ignoring the new value "
                        << optarg << ".\n";
            else if (optarg)
              {
                std::cerr << "option " << opt_name << " with value " << optarg;
                bool success = info.parse(optarg);
                if (success)
                  std::cerr << " set\n";
                else
                  std::cerr << " failed\n";
              }
            else
              info.parse("");

            info.matched = true;
          }
        else
          return false;
      }
    if (optind < argc)
      {
        std::cerr << "non-option argv-elements: ";
        while (optind < argc)
          std::cerr << argv[optind++] << '\n';
      }
    return true;
  }

  template <>
  inline bool
  set(std::string &var, const std::string &optarg)
  {
    var = optarg;
    return true;
  }

  template <>
  inline bool
  set(bool &var, const std::string &optarg)
  {
    std::istringstream ss(optarg);
    ss >> var;
    if (ss.fail())
      {
        ss.clear();
        ss >> std::boolalpha >> var;
      }
    return !ss.fail();
  }

  template <typename T>
  bool
  set_vector(T &var, const std::string &, std::istringstream &ss)
  {
    var.clear();
    typename T::value_type temp;
    while (ss >> temp)
      {
        var.push_back(std::move(temp));
        if (ss.fail())
          return false;
      }
    return true;
  }

  template <typename T>
  bool
  set(T &var, const std::string &optarg)
  {
    std::istringstream ss(optarg);
    if constexpr (is_optional<T>::value)
      {
        typename T::value_type temp;
        var.emplace();
        return set(*var, optarg);
      }
    else if constexpr (is_iterable<T>::value)
      {
        if constexpr (is_vector<T>::value)
          return set_vector(var, optarg, ss);
        else
          return std::all_of(std::begin(var), std::end(var), [&](auto &el) {
            return (ss >> el) ? true : false;
          });
      }
    else
      return (ss >> var) ? !ss.fail() : false;
  }

  inline option_map
  get_option_map()
  {
    option_map_ config_;
    config_['h']      = false;
    option_map config = std::make_shared<util::option_map_>(config_);
    return config;
  }

  template <typename T>
  T
  option_acc::at(char key) const
  {
    return std::any_cast<T>(opts->at(key));
  }

  template <typename T>
  T
  option_acc::unsafe_at(char key) const
  {
    return std::any_cast<T>((*opts)[key]);
  }
} // namespace util
