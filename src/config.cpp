#include "config.h"
#include <fstream>
#include <sstream>
#include <exception>
#include "glog/logging.h"

using namespace std;

AnyConversion EmptyAnyConversion("");

Config::Config() {
}
void Config::ValueCheck(bool condition, string const& key){
    if(!condition)
        throw std::runtime_error(key + " not found");
}
bool Config::Load(string const &configFilePath) {
    LOG(INFO) << "Load config file: " << configFilePath << endl;
    if (configFilePath.find(JSON_FILE_POSTFIX) != string::npos) {
        loadJson(configFilePath);
    } else if (configFilePath.find(INI_FILE_POSTFIX) != string::npos) {
        loadText(configFilePath);
        string newConfigFilePath = convertTextToJson(configFilePath);
        LOG(WARNING)
            << "Old config file detected, new config has been created: "
            << newConfigFilePath << endl;
        Load(newConfigFilePath);
    } else {
        LOG(ERROR) << "Cannot find config file:" << configFilePath.c_str()
            << endl;
        exit(1);
    }

    return true;
}

bool Config::KeyExist(string const &section, string const &entry) {
    std::map<string, AnyConversion>::const_iterator ci = content_.find(
        section + '/' + entry);
    return !(ci == content_.end());
}

void Config::AddEntry(string key, AnyConversion value) {
    content_[key] = value;
}

AnyConversion const &Config::Value(string const &key) const {

    std::map<string, AnyConversion>::const_iterator ci = content_.find(key);
    if (ci == content_.end()) {

        return EmptyAnyConversion;
    }
    return ci->second;
}

AnyConversion const &Config::Value(const char *keyFormat, int index) const {

    char key[256];
    sprintf(key, keyFormat, index);
    return Value(string(key));
}
bool Config::LoadString(string const &data) {
    Json::Value root;
    Json::Reader reader;
    bool parse_success = reader.parse(data, root, false);
    if (!parse_success) {
        return false;
    }
    parseJsonNode(root, "");
    return true;
}
vector<AnyConversion> Config::getArray(string const & key) {
    vector<AnyConversion> result;
        auto array_size_value = Value(key + "/Size");
    ValueCheck(!array_size_value.empty(), key);
    
    int array_size = static_cast<int>(array_size_value);
    for (int i = 0; i < array_size; i++) {
        string itemkey = key + std::to_string(i);
        result.push_back(Value(itemkey));
    }
    return result;
}

vector<float> Config::getFloatArray(string const & key) {
    vector<AnyConversion> anyset = getArray(key);
    vector<float> result;
    for (int i = 0; i < anyset.size(); i++)
        result.push_back(static_cast<float>(anyset[i]));
    return result;
}
vector<int> Config::getIntArray(string const & key) {
    vector<AnyConversion> anyset = getArray(key);
    vector<int> result;
    for (int i = 0; i < anyset.size(); i++)

        result.push_back(static_cast<int>(anyset[i]));
    return result;
}
vector<string> Config::getStringArray(string const & key) {
    vector<AnyConversion> anyset = getArray(key);
    vector<string> result;
    for (int i = 0; i < anyset.size(); i++)
        result.push_back(static_cast<string>(anyset[i]));
    return result;
}
float Config::getFloat(string const & key) {
    auto value = Value(key);
    ValueCheck(!value.empty(), key);
    return static_cast<float>(value);
}
int Config::getInteger(string const & key) {
        auto value = Value(key);
    ValueCheck(!value.empty(), key);
    return static_cast<int>(value);
}
string Config::getString(string const & key) {
        auto value = Value(key);
    ValueCheck(!value.empty(), key);
    return static_cast<string>(value);
}
bool Config::loadText(string const &configFile) {
    FILE *file = fopen(configFile.c_str(), "r");

    if (!file || feof(file))
        return false;

    char buf[1024];
    string line;
    string name;
    string val;
    string inSection;
    int posEqual;
    while (fgets(buf, sizeof(buf), file)) {
        line = buf;
        if (line.length() < 3)
            continue;

        if (line[0] == '#')
            continue;
        if (line[0] == ';')
            continue;
        if (line[0] == '[') {
            inSection = trim(line.substr(1, line.find(']') - 1));
            continue;
        }

        // only space and tab are allowed to separate variable name and value
        posEqual = line.find(INI_SEPERATOR);

        name = trim(line.substr(0, posEqual));
        val = trim(line.substr(posEqual + 1));

        string key = inSection + '/' + name;
        AddEntry(key, AnyConversion(val));
    }
    fclose(file);
    return true;
}

bool Config::loadJson(string const &configFile) {
    // parse the json config file
    Json::Value root;
    Json::Reader reader;
    ifstream ifs(configFile, ifstream::binary);
    bool parse_success = reader.parse(ifs, root, false);
    if (!parse_success) {
        fprintf(stderr, "Parse %s failed.\n", configFile.c_str());
        return false;
    }
    parseJsonNode(root, "");
    return true;
}

void Config::parseJsonNode(Json::Value &node, const string prefix) {
    Json::Value::Members::iterator itr;

    Json::Value::Members children = node.getMemberNames();

    for (itr = children.begin(); itr != children.end(); ++itr) {
        string sectionName = *itr;
        Json::Value section = node[sectionName];
        string key = prefix + '/' + sectionName;
        if (prefix == "") {
            key = sectionName;
        }

        if (section.isObject()) {
            string newPrefix = prefix + '/' + sectionName;
            if (prefix == "") {
                newPrefix = sectionName;
            }
            parseJsonNode(section, newPrefix);
        } else if (section.isArray()) {
            int arraySize = section.size();
            AddEntry(key + "/Size", AnyConversion(arraySize));
            for (int i = 0; i < arraySize; ++i) {
                string newPrefix = prefix + '/' + sectionName;
                if (prefix == "") {
                    newPrefix = sectionName;
                }
                newPrefix = newPrefix + std::to_string(i);
                if (section[i].isArray() || section[i].isObject())
                    parseJsonNode(section[i], newPrefix);
                else {
                    if (section[i].isInt()) {
                        AddEntry(newPrefix, AnyConversion(section[i].asInt()));
                    } else if (section[i].isNumeric()) {
                        AddEntry(newPrefix, AnyConversion(section[i].asDouble()));
                    } else {
                        AddEntry(newPrefix, AnyConversion(section[i].asString()));
                    }
                }
            }
        } else if (section.isInt()) {
            AddEntry(key, AnyConversion(section.asInt()));
        } else if (section.isNumeric()) {
            AddEntry(key, AnyConversion(section.asDouble()));
        } else {
            AddEntry(key, AnyConversion(section.asString()));
        }
    }
}

string Config::convertTextToJson(string const &configFile) {
// declare json writer and root node
    Json::FastWriter fast_writer;
    Json::Value root;
    string pre_section_name;
    Json::Value section;
// construct json file
    for (map<string, AnyConversion>::iterator it = content_.begin();
         it != content_.end(); ++it) {
        printf("%s %s\n", it->first.c_str(),
               static_cast<string>(it->second).c_str());
        // get current section name
        string cur_section_name;
        size_t section_pos = it->first.find("/");
        cur_section_name = it->first.substr(0, section_pos);
        size_t space_pos = it->first.find("\t");
        if (space_pos == string::npos) {
            space_pos = it->first.find("\b");
        }
        // get current key
        string cur_key = it->first.substr(section_pos + 1, space_pos);
        // get current value
        string cur_value = it->second;
        // construct section and root
        if (pre_section_name.compare(cur_section_name) != 0) {  // new section begin
            if (!pre_section_name.empty()) {  // put finished pre section to root
                root[pre_section_name] = section;
                section.clear();
            }
            // update pre_section_name
            pre_section_name = cur_section_name;
        }
        // inner same section, add key-value item
        if (isControlFlag(cur_value)) {
            section[cur_key] = atoi(cur_value.c_str());
        } else {
            section[cur_key] = cur_value;
        }
    }
// add last section to root
    root[pre_section_name] = section;
// save converted config file
    size_t pos = configFile.rfind(".");
    string config_loc = configFile.substr(0, pos);
    config_loc += JSON_FILE_POSTFIX;
    string outstring = fast_writer.write(root);
    FILE *fp = fopen(config_loc.c_str(), "w+");
    fwrite(outstring.c_str(), 1, outstring.size(), fp);
    fclose(fp);
    return config_loc;
}

bool Config::isControlFlag(string const &str) {
    if (str.compare("0") == 0 || str.compare("1") == 0) {
        return true;
    } else {
        return false;
    }
}

void Config::DumpValues() {

    for (std::map<string, AnyConversion>::const_iterator it = content_.begin();
         it != content_.end(); it++) {
        std::cerr << (string) it->first << " = " << (string) it->second
            << std::endl;
    }
}

string Config::trim(string const &source, char const *delims) {
    string result(source);
    string::size_type index = result.find_last_not_of(delims);
    if (index != string::npos)
        result.erase(++index);

    index = result.find_first_not_of(delims);
    if (index != string::npos)
        result.erase(0, index);
    else
        result.erase();
    return result;
}

void Config::Clear() {
    content_.clear();
}