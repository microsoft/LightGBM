# coding: utf-8
"""This script generates LightGBM/src/io/config_auto.cpp file
with list of all parameters, aliases table and other routines
along with parameters description in LightGBM/docs/Parameters.rst file
from the information in LightGBM/include/LightGBM/config.h file.
"""
import os


def GetParameterInfos(config_hpp):
    is_inparameter = False
    parameter_group = None
    cur_key = None
    cur_info = {}
    keys = []
    member_infos = []
    with open(config_hpp) as config_hpp_file:
        for line in config_hpp_file:
            if "#pragma region Parameters" in line:
                is_inparameter = True
            elif "#pragma region" in line and "Parameters" in line:
                cur_key = line.split("region")[1].strip()
                keys.append(cur_key)
                member_infos.append([])
            elif '#pragma endregion' in line:
                if cur_key is not None:
                    cur_key = None
                elif is_inparameter:
                    is_inparameter = False
            elif cur_key is not None:
                line = line.strip()
                if line.startswith("//"):
                    key, _, val = line[2:].partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key not in cur_info:
                        if key == "descl2" and "desc" not in cur_info:
                            cur_info["desc"] = []
                        elif key != "descl2":
                            cur_info[key] = []
                    if key == "desc":
                        cur_info["desc"].append(("l1", val))
                    elif key == "descl2":
                        cur_info["desc"].append(("l2", val))
                    else:
                        cur_info[key].append(val)
                elif line:
                    has_eqsgn = False
                    tokens = line.split("=")
                    if len(tokens) == 2:
                        if "default" not in cur_info:
                            cur_info["default"] = [tokens[1][:-1].strip()]
                        has_eqsgn = True
                    tokens = line.split()
                    cur_info["inner_type"] = [tokens[0].strip()]
                    if "name" not in cur_info:
                        if has_eqsgn:
                            cur_info["name"] = [tokens[1].strip()]
                        else:
                            cur_info["name"] = [tokens[1][:-1].strip()]
                    member_infos[-1].append(cur_info)
                    cur_info = {}
    return keys, member_infos


def GetNames(infos):
    names = []
    for x in infos:
        for y in x:
            names.append(y["name"][0])
    return names


def GetAlias(infos):
    pairs = []
    for x in infos:
        for y in x:
            if "alias" in y:
                name = y["name"][0]
                alias = y["alias"][0].split(',')
                for name2 in alias:
                    pairs.append((name2.strip(), name))
    return pairs


def SetOneVarFromString(name, param_type, checks):
    ret = ""
    univar_mapper = {"int": "GetInt", "double": "GetDouble", "bool": "GetBool", "std::string": "GetString"}
    if "vector" not in param_type:
        ret += "  %s(params, \"%s\", &%s);\n" % (univar_mapper[param_type], name, name)
        if len(checks) > 0:
            for check in checks:
                ret += "  CHECK(%s %s);\n" % (name, check)
        ret += "\n"
    else:
        ret += "  if (GetString(params, \"%s\", &tmp_str)) {\n" % (name)
        type2 = param_type.split("<")[1][:-1]
        if type2 == "std::string":
            ret += "    %s = Common::Split(tmp_str.c_str(), ',');\n" % (name)
        else:
            ret += "    %s = Common::StringToArray<%s>(tmp_str, ',');\n" % (name, type2)
        ret += "  }\n\n"
    return ret


def GenParameterDescription(sections, descriptions, params_rst):

    def parse_check(check, reverse=False):
        try:
            idx = 1
            float(check[idx:])
        except ValueError:
            idx = 2
            float(check[idx:])
        if reverse:
            reversed_sign = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}
            return check[idx:], reversed_sign[check[:idx]]
        else:
            return check[idx:], check[:idx]

    params_to_write = []
    for section_name, section_params in zip(sections, descriptions):
        params_to_write.append('{0}\n{1}'.format(section_name, '-' * len(section_name)))
        for param_desc in section_params:
            name = param_desc['name'][0]
            default_raw = param_desc['default'][0]
            default = default_raw.strip('"') if len(default_raw.strip('"')) > 0 else default_raw
            param_type = param_desc.get('type', param_desc['inner_type'])[0].split(':')[-1].split('<')[-1].strip('>')
            options = param_desc.get('options', [])
            if len(options) > 0:
                options_str = ', options: ``{0}``'.format('``, ``'.join([x.strip() for x in options[0].split(',')]))
            else:
                options_str = ''
            aliases = param_desc.get('alias', [])
            if len(aliases) > 0:
                aliases_str = ', aliases: ``{0}``'.format('``, ``'.join([x.strip() for x in aliases[0].split(',')]))
            else:
                aliases_str = ''
            checks = sorted(param_desc.get('check', []))
            checks_len = len(checks)
            if checks_len > 1:
                number1, sign1 = parse_check(checks[0])
                number2, sign2 = parse_check(checks[1], reverse=True)
                checks_str = ', constraints: ``{0} {1} {2} {3} {4}``'.format(number2, sign2, name, sign1, number1)
            elif checks_len == 1:
                number, sign = parse_check(checks[0])
                checks_str = ', constraints: ``{0} {1} {2}``'.format(name, sign, number)
            else:
                checks_str = ''
            main_desc = '-  ``{0}`` :raw-html:`<a id="{0}" title="Permalink to this parameter" href="#{0}">&#x1F517;&#xFE0E;</a>`, default = ``{1}``, type = {2}{3}{4}{5}'.format(name, default, param_type, options_str, aliases_str, checks_str)
            params_to_write.append(main_desc)
            params_to_write.extend([' ' * 3 * int(desc[0][-1]) + '-  ' + desc[1] for desc in param_desc['desc']])

    with open(params_rst) as original_params_file:
        all_lines = original_params_file.read()
        before, start_sep, _ = all_lines.partition('.. start params list\n\n')
        _, end_sep, after = all_lines.partition('\n\n.. end params list')

    with open(params_rst, "w") as new_params_file:
        new_params_file.write(before)
        new_params_file.write(start_sep)
        new_params_file.write('\n\n'.join(params_to_write))
        new_params_file.write(end_sep)
        new_params_file.write(after)


def GenParameterCode(config_hpp, config_out_cpp):
    keys, infos = GetParameterInfos(config_hpp)
    names = GetNames(infos)
    alias = GetAlias(infos)
    str_to_write = "/// This file is auto generated by LightGBM\\helper\\parameter_generator.py from LightGBM\\include\\LightGBM\\config.h file.\n"
    str_to_write += "#include<LightGBM/config.h>\nnamespace LightGBM {\n"
    # alias table
    str_to_write += "std::unordered_map<std::string, std::string> Config::alias_table({\n"
    for pair in alias:
        str_to_write += "  {\"%s\", \"%s\"},\n" % (pair[0], pair[1])
    str_to_write += "});\n\n"
    # names
    str_to_write += "std::unordered_set<std::string> Config::parameter_set({\n"
    for name in names:
        str_to_write += "  \"%s\",\n" % (name)
    str_to_write += "});\n\n"
    # from strings
    str_to_write += "void Config::GetMembersFromString(const std::unordered_map<std::string, std::string>& params) {\n"
    str_to_write += "  std::string tmp_str = \"\";\n"
    for x in infos:
        for y in x:
            if "[doc-only]" in y:
                continue
            param_type = y["inner_type"][0]
            name = y["name"][0]
            checks = []
            if "check" in y:
                checks = y["check"]
            tmp = SetOneVarFromString(name, param_type, checks)
            str_to_write += tmp
    # tails
    str_to_write += "}\n\n"
    str_to_write += "std::string Config::SaveMembersToString() const {\n"
    str_to_write += "  std::stringstream str_buf;\n"
    for x in infos:
        for y in x:
            if "[doc-only]" in y:
                continue
            param_type = y["inner_type"][0]
            name = y["name"][0]
            if "vector" in param_type:
                if "int8" in param_type:
                    str_to_write += "  str_buf << \"[%s: \" << Common::Join(Common::ArrayCast<int8_t, int>(%s),\",\") << \"]\\n\";\n" % (name, name)
                else:
                    str_to_write += "  str_buf << \"[%s: \" << Common::Join(%s,\",\") << \"]\\n\";\n" % (name, name)
            else:
                str_to_write += "  str_buf << \"[%s: \" << %s << \"]\\n\";\n" % (name, name)
    # tails
    str_to_write += "  return str_buf.str();\n"
    str_to_write += "}\n\n"
    str_to_write += "}\n"
    with open(config_out_cpp, "w") as config_out_cpp_file:
        config_out_cpp_file.write(str_to_write)

    return keys, infos


if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(__file__))
    config_hpp = os.path.join(current_dir, os.path.pardir, 'include', 'LightGBM', 'config.h')
    config_out_cpp = os.path.join(current_dir, os.path.pardir, 'src', 'io', 'config_auto.cpp')
    params_rst = os.path.join(current_dir, os.path.pardir, 'docs', 'Parameters.rst')
    sections, descriptions = GenParameterCode(config_hpp, config_out_cpp)
    GenParameterDescription(sections, descriptions, params_rst)
