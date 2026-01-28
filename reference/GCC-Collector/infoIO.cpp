#include "types.h"
#include "infoIO.h"
#include "camera.h"
#include "setLevel.h"
#include <fstream>
#include <string>
#include <io.h>
#include <filesystem>
#include <sstream>
#include <vector>
#include <Windows.h>
#include <chrono>

void set_status_text(std::string text)
{
	UI::_SET_NOTIFICATION_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)text.c_str());
	UI::_DRAW_NOTIFICATION(1, 1);
}

void log_to_pedTxt(std::string message, char *fileName) {
	std::ofstream logfile(fileName, std::ios_base::app);
	logfile << message + "\n";
	logfile.close();
}

void log_to_file(std::string msg)
{
	char *logfile = ".\\data\\GCC-CL.log";
	std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
		std::chrono::system_clock::now().time_since_epoch()
		);
	char sec[20];
	sprintf(sec, "[%I64d] : ", ms);
	msg = std::string(sec) + msg;
	log_to_pedTxt(msg, logfile);
}

bool initDataDir()
{
	LPCSTR dataroot = "data";
	// return GetFileAttributesA(dataroot) == dirMark || CreateDirectory(dataroot, NULL);
	if (GetFileAttributesA(dataroot) != dirMark) {
		if (CreateDirectory(dataroot, NULL)) {
			log_to_file("create data dir");
			return true;
		}
		else {
			log_to_file("create data dir failed");
			return false;
		}
	}
	else {
		log_to_file("data dir already exist");
		return true;
	}
}

char nowFold[fileLength] = "data\\";
char nowFolds[4][fileLength] = { "data\\", "data\\", "data\\", "data\\" };

std::vector<int> parts, subParts, imageNums;
bool InitPartNo()
{
    // Clear previous data
    parts.clear();
    subParts.clear();
    imageNums.clear();

    std::string root_dir_str = ".\\data\\"; // Use std::string for root directory path

    // Use std::filesystem::path to avoid string concatenation issues and better path handling
    std::filesystem::path root_dir(root_dir_str);

    // Check if the root directory exists and is accessible
    if (!std::filesystem::exists(root_dir)) {
        log_to_file("Error: Data root directory does not exist: " + root_dir_str);
        return false; // Directory does not exist, return failure directly
    }
    if (!std::filesystem::is_directory(root_dir)) {
        log_to_file("Error: Data root directory is not a directory: " + root_dir_str);
        return false; // Not a directory, return failure directly
    }

    try {
        // Iterate through all entries in the root directory
        for (const auto& entry : std::filesystem::directory_iterator(root_dir)) {
            // Get the full path of the current entry
            std::filesystem::path current_path = entry.path();
            std::string subFold = current_path.string(); // Convert path to string

            // Check if the current entry is a directory and contains "part_"
            // Better to directly check the directory name rather than the full path string
            std::string folder_name = current_path.filename().string(); // Get directory name
            int st = folder_name.find("part_");

            if (st != std::string::npos && std::filesystem::is_directory(current_path)) { // Ensure it's a directory
                st += 5; // Skip "part_"

                // Extract parts number
                // Find the next '_' to determine the end of X in part_X
                size_t underscore_pos = folder_name.find("_", st);
                std::string part_str;
                if (underscore_pos != std::string::npos) {
                    part_str = folder_name.substr(st, underscore_pos - st);
                } else {
                    // If no underscore, the format is part_X
                    part_str = folder_name.substr(st);
                }

                try {
                    int part_num = std::stoi(part_str);
                    parts.push_back(part_num);
                } catch (const std::invalid_argument& e) {
                    log_to_file("Error: Invalid argument when converting part number: " + part_str + " in " + folder_name + " - " + e.what());
                    continue; // Skip current directory
                } catch (const std::out_of_range& e) {
                    log_to_file("Error: Out of range when converting part number: " + part_str + " in " + folder_name + " - " + e.what());
                    continue; // Skip current directory
                }

                // Extract subParts number (assuming it's always the last digit, and there's an underscore separator)
                int sub_part_num = 0;
                if (underscore_pos != std::string::npos && folder_name.length() > underscore_pos + 1) {
                     // Check if the last character is a digit
                    if (std::isdigit(folder_name.back())) {
                        try {
                            sub_part_num = std::stoi(folder_name.substr(folder_name.length() - 1));
                        } catch (const std::invalid_argument& e) {
                            log_to_file("Warning: Invalid argument when converting sub-part number: " + folder_name.substr(folder_name.length() - 1) + " in " + folder_name + " - " + e.what());
                        } catch (const std::out_of_range& e) {
                            log_to_file("Warning: Out of range when converting sub-part number: " + folder_name.substr(folder_name.length() - 1) + " in " + folder_name + " - " + e.what());
                        }
                    }
                }
                subParts.push_back(sub_part_num);


                std::string imgFold; // Image folder path for internal loop
                int thisImgNum = 0;

                log_to_file("Iterating sub-directory: " + subFold);

                // Iterate through the contents of the current part_X_Y directory
                try {
                    for (const auto& sub_entry : std::filesystem::directory_iterator(current_path)) { // Use current_path for iteration
                        std::filesystem::path img_path = sub_entry.path();
                        imgFold = img_path.string(); // Get the full path string

                        log_to_file("  Checking file/directory: " + imgFold);

                        // Check if it's a directory, and if so, count it
                        // Use std::filesystem::is_directory for modern and type-safe check
                        if (std::filesystem::is_directory(img_path)) {
                            thisImgNum++;
                        }
                        // If you need to check GetFileAttributesA, ensure dirMark is correctly defined as FILE_ATTRIBUTE_DIRECTORY
                        // if(GetFileAttributesA(imgFold.c_str()) == dirMark) thisImgNum ++;
                    }
                } catch (const std::filesystem::filesystem_error& e) {
                    log_to_file("Error: Filesystem error while iterating sub-directory (" + subFold + "): " + e.what());
                    // Even if internal directory iteration fails, we still want to record external part information, so don't return false here
                }

                imageNums.push_back(thisImgNum); // Add image count

                // Log parsing results
                log_to_file(std::to_string(parts.back()) + " " + std::to_string(subParts.back()) + " " + std::to_string(thisImgNum));

            } // end of if (st != std::string::npos && std::filesystem::is_directory(current_path))
        } // end of for (const auto& entry : std::filesystem::directory_iterator(root_dir))
    } catch (const std::filesystem::filesystem_error& e) {
        log_to_file("Fatal error: Filesystem error while iterating root directory (" + root_dir_str + "): " + e.what());
        return false; // Root directory iteration failed, function returns false
    } catch (const std::exception& e) {
        log_to_file("Fatal error: Unknown exception occurred: " + std::string(e.what()));
        return false;
    }

    return true;
}

int workId = -1;
int defaultFold()
{
	int imgNum = readImgNum(), sceneNum = parts.size();
	if(workId == -1) {
		for(workId = 0; workId < sceneNum; workId++) {
			if(imageNums[workId] < imgNum) break;
		}
		if(workId >= sceneNum) return -1;
	}

	std::string foldName = "part_" + std::to_string(parts[workId]) + "_" + std::to_string(subParts[workId]);
	strcpy(nowFold, "data\\");
	strcat(nowFold, foldName.c_str());
	log_to_file("deault fold is " + std::string(foldName));
	return imageNums[workId];
}

void markAddOneImage() {
	int thisImgNum = ++ imageNums[workId];
	if(readImgNum() == thisImgNum) workId = -1;
}

void createNewFold()
{
	std::string root_dir = ".\\data\\", subFold;
	bool exist[partLength] = {false};
	for (auto & p : std::filesystem::directory_iterator(root_dir)) {
		std::stringstream conv;
		conv << p; conv >> subFold;
		int st = subFold.find("part_");
		if (st != -1) {
			st += 5;

			int subLen = subFold.length() - 2 - st;
			if (subFold.find("_", st + 5) == -1) subLen += 2;
			int foldNo = std::stoi(subFold.substr(st, subLen));
			exist[foldNo] = true;
		}
	}

	auto createFoldExe = [](int part) {
		for (int subpart = 0; subpart < 4; subpart++) {
			strcpy(nowFolds[subpart], "data\\");
			strcat(nowFolds[subpart], ("part_" + std::to_string(part)).c_str());
			strcat(nowFolds[subpart], ("_" + std::to_string(subpart)).c_str());
			CreateDirectory(nowFolds[subpart], NULL);
			log_to_file("create new fold = " + std::string(nowFold));
		}
	};

	for (int i = partLength - 1; i > 0; i--) {
		if (exist[i]) {
			i = i + 1;
			createFoldExe(i);
			strcpy(nowFold, nowFolds[0]);
			return;
		}
	}
	createFoldExe(1);
}

void changeFoldNo(int No)
{
	strcpy(nowFold, nowFolds[No]);
}

void foldCat(char *subString, char *useFold)
{
	char newString[fileLength] = "\\";
	strcat(newString, subString);
	strcpy(subString, useFold);
	strcat(subString, newString);
}

void foldCat(char *subString)
{
	foldCat(subString, nowFold);
}

void foldCat(WCHAR *substring, char *useFold)
{
	WCHAR newString[fileLength];
	swprintf(newString, fileLength, L"%hs\\", useFold);
	wcscat(newString, substring);
	wcscpy(substring, newString);
}

 void foldCat(WCHAR *substring)
 {
 	foldCat(substring, nowFold);
 }

void writeCamInfo(const Vector3 &loc, const Vector3 &rot, const float &fov, int foldNo)
{
	char eyeInfoFile[fileLength] = "eyeInfo.log";
	foldCat(eyeInfoFile, nowFolds[foldNo]);
	std::ofstream info(eyeInfoFile);
	info << loc.x << " " << loc.y << " " << loc.z << std::endl;
	info << rot.x << " " << rot.y << " " << rot.z << std::endl;
	info << fov;
	info.close();
}

void writeCamInfo(const Vector3 &loc, const Vector3 &rot, const float &fov)
{
	writeCamInfo(loc, rot, fov, 0);
}

void wriet4Camera()
{
	char cpNowFold[fileLength];
	strcpy(cpNowFold, nowFold);
	strcpy(nowFold, nowFolds[0]);
	Vector3 loc, rot; float fov;
	readCamInfo(loc, rot, fov);

	float mix, mxx, miy, mxy;
	readAreaBorder(mix, mxx, miy, mxy);
	float cx = (mxx + mix) / 2;
	float cy = (mxy + miy) / 2;
	log_to_file(std::to_string(cx) + " " + std::to_string(cy));

	for (int i = 0; i < 4; i++) {
		writeCamInfo(loc, rot, fov, i);

		float nx = -(loc.y - cy) + cx;
		float ny = cy + loc.x - cx;
		loc.x = nx, loc.y = ny;

		float &r = rot.z;
		r += 90;
		if (r > 180) r - 180 - 180;
	}
	strcpy(nowFold, cpNowFold);
}

void readCamInfo(float &locx, float &locy, float &locz, float rotx, float &roty, float &rotz, float &fov)
{
	char eyeInfoFile[fileLength] = "eyeInfo.log";
	foldCat(eyeInfoFile);
	std::ifstream camInput(eyeInfoFile);
	camInput >> locx >> locy >> locz;
	camInput >> rotx >> roty >> rotz;
	camInput >> fov;
	camInput.close();
}


void readCamInfo(Vector3 &loc, Vector3 &rot, float &fov)
{
	char eyeInfoFile[fileLength] = "eyeInfo.log";
	foldCat(eyeInfoFile);
	log_to_file(eyeInfoFile);
	std::ifstream camInput(eyeInfoFile);
	camInput >> loc.x >> loc.y >> loc.z;
	camInput >> rot.x >> rot.y >> rot.z;
	camInput >> fov;
	camInput.close();
}

void writeAreaInfo(const int &n)
{
	for (int i = 0; i < 4; i++) {
		char areaInfoFile[fileLength] = "areaInfo.log";
		foldCat(areaInfoFile, nowFolds[i]);
		log_to_file(areaInfoFile);
		std::ofstream info(areaInfoFile);
		info << n << std::endl;
		info.close();
	}
}

void writeAreaInfo(const Vector3 &loc)
{
	for (int i = 0; i < 4; i++) {
		char areaInfoFile[fileLength] = "areaInfo.log";
		foldCat(areaInfoFile, nowFolds[i]);
		std::ofstream info(areaInfoFile, std::ios_base::app);
		info << loc.x << ' ' << loc.y << std::endl;
		info.close();
	}
}

void writeZheight(float z)
{
	for (int i = 0; i < 4; i++) {
		char areaInfoFile[fileLength] = "Zheight.log";
		foldCat(areaInfoFile, nowFolds[i]);
		std::ofstream info(areaInfoFile);
		info << z << std::endl;
		info.close();
	}
}


void readAreaInfo(int &n, std::vector<pedLocation> &pedLocations)
{
	char areaInfoFile[fileLength] = "areaInfo.log";
	foldCat(areaInfoFile);
	std::ifstream info(areaInfoFile);
	pedLocation po;
	info >> n;
	for (int i = 0; i < n; i++) {
		info >> po.x >> po.y;
		pedLocations.push_back(po);
	}
	info.close();
}

float readZheight()
{
	float z;
	char areaInfoFile[fileLength] = "Zheight.log";
	foldCat(areaInfoFile);
	std::ifstream info(areaInfoFile);
	info >> z;
	info.close();

	return z;
}

void readAreaBorder(float &mix, float &mxx, float &miy, float &mxy)
{
	char areaInfoFile[fileLength] = "areaInfo.log";
	foldCat(areaInfoFile);
	log_to_file(areaInfoFile);
	std::ifstream circle(areaInfoFile);
	int n; float x, y;
	bool first = true;
	while (circle >> n) {
		while (n--) {
			circle >> x >> y;
			if (first) {
				mix = mxx = x, miy = mxy = y;
				first = false;
			}
			else {
				mix = min(x, mix), mxx = max(x, mxx);
				miy = min(y, miy), mxy = max(y, mxy);
			}
		}
	}
	circle.close();
}

bool fileExist(char *path)
{
	FILE* fh = fopen(path, "r");
	return fh != NULL;
}

bool fileExist(WCHAR *path)
{
	char cpath[fileLength];
	sprintf(cpath, "%ws", path);
	return fileExist(cpath);
}

void writeLeveFile(int level)
{
	for (int i = 0; i < 4; i++) {
		char levelFile[fileLength] = "levelInfo.log";
		foldCat(levelFile, nowFolds[i]);
		std::ofstream levelInfo(levelFile);
		levelInfo << level;
		levelInfo.close();
	}
}

int readLevelFile()
{
	char levelfile[fileLength] = "levelInfo.log";
	foldCat(levelfile);
	std::ifstream info(levelfile);
	int levNo;
	info >> levNo;
	info.close();
	return level[levNo].maxNum;
}

int readImgNum()
{
	int imgNum = 5;
	char* imgNumFile = "imageNum.txt";
	if(fileExist(imgNumFile)) {
		std::ifstream imgf(imgNumFile);
		imgf >> imgNum;
		imgf.close();
	}
	return imgNum;
}
