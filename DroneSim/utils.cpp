/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com
			(C) Alexander Blade 2015
*/

#include "utils.h"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include "script.h"

extern "C" IMAGE_DOS_HEADER __ImageBase; // MSVC specific, with other compilers use HMODULE from DllMain

void setStatusText(std::string text)
{
	UI::_SET_NOTIFICATION_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)text.c_str());
	UI::_DRAW_NOTIFICATION(1, 1);
}

void log_to_pedTxt(std::string text, char* file)
{
	FILE* fp;
	fopen_s(&fp, file, "a");
	if (fp == NULL) {
		return;
	}
	fprintf(fp, "%s\n", text.c_str());
	fclose(fp);
}

std::string cachedModulePath;

std::string GetCurrentModulePath()
{
	if (cachedModulePath.empty())
	{
		// get module path
		char modPath[MAX_PATH];
		memset(modPath, 0, sizeof(modPath));
		GetModuleFileNameA((HMODULE)&__ImageBase, modPath, sizeof(modPath));
		for (size_t i = strlen(modPath); i > 0; i--)
		{
			if (modPath[i - 1] == '\\')
			{
				modPath[i] = 0;
				break;
			}
		}
		cachedModulePath = modPath;
	}
	return cachedModulePath;
}

std::string roundNumber(float number)
{
	std::ostringstream out;
	out << std::setprecision(1) << std::fixed << std::showpoint << number;
	std::string roundResult = out.str();
	return roundResult;
}

float actionInputFloat()
{
	GAMEPLAY::DISPLAY_ONSCREEN_KEYBOARD(true, "FMMC_KEY_TIP8", "", "", "", "", "", 6);

	while (GAMEPLAY::UPDATE_ONSCREEN_KEYBOARD() == 0) {
		WAIT(0);
	}


	if (GAMEPLAY::IS_STRING_NULL_OR_EMPTY(GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT())) {
		return -1.0f;
	}
	char * keyboardValue = GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT();
	std::string strValue = std::string(keyboardValue);
	return atof(keyboardValue);
}


Vector3 MathUtils::rotationToDirection(Vector3 rotation)
{
	//big thanks to camxxcore's C# code https://github.com/CamxxCore/ScriptCamTool/blob/master/GTAV_ScriptCamTool/Utils.cs
	float retZ = rotation.z * 0.01745329f;
	float retX = rotation.x * 0.01745329f;
	float absX = abs(cos(retX));
	Vector3 retVector = { 0.0,0.0,0.0 };
	retVector.x = (float)-(sin(retZ) * absX);
	retVector.y = (float)cos(retZ) * absX;
	retVector.z = (float)sin(retX);
	return retVector;
}

Vector3 MathUtils::crossProduct(Vector3 a, Vector3 b)
{
	//http://onlinemschool.com/math/assistance/vector/multiply1/
	Vector3 retVector = { 0.0,0.0,0.0 };
	retVector.x = a.y*b.z - a.z*b.y;
	retVector.y = a.z*b.x - a.x*b.z;
	retVector.z = a.x*b.y - a.y*b.x;
	return retVector;
}


void StringUtils::split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
}


std::vector<std::string> StringUtils::split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}