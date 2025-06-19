workspace "walking-stick"
	architecture "x64"
	
	configurations
	{
		"debug",
		"release",
		"profile",
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "thirdparty/ImGui"
include "thirdparty/GLFW"
include "thirdparty/glad"

project "walking-stick"
	location ""
	kind "ConsoleApp"
	language "c++"
	cppdialect "c++latest"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"source/**.h",
		"source/**.cpp",
		"source/**.hpp",
	}

	includedirs
	{
		"source",
		"thirdparty/GLFW/include",
		"thirdparty/glad/include",
		"thirdparty/imgui",
		"thirdparty/spdlog/include",
		"thirdparty/sfml/include",
		"thirdparty/glm",
		"thirdparty/entt/single_include",
	}

	libdirs
	{ 
		"thirdparty/sfml/lib",
	}

	links
	{
		"GLFW",
		"glad",
		"ImGui",
		"opengl32",
		"winmm",
		"gdi32",
		"user32",
		"advapi32"
	}

	defines
	{
		"_CRT_SECURE_NO_WARNINGS",
		"GLFW_INCLUDE_NONE",
		"SFML_STATIC",
	}

	ignoredefaultlibraries 
	{ 
		"libcmt.lib"
	}

	filter "system:windows"
		systemversion "latest"

		defines
		{
			"PLATFORM_WINDOWS",
		}

	filter "configurations:debug"
		links 
		{
			"sfml-graphics-s-d",
			"sfml-window-s-d",
			"sfml-system-s-d"
		}
		
		defines {"DEBUG"}
		runtime "Debug"
		symbols "on"
	
	filter "configurations:release"
		links 
		{
			"sfml-graphics-s",
			"sfml-window-s",
			"sfml-system-s"
		}

		defines {"RELEASE"}
		runtime "Release"
		optimize "on"

	filter "configurations:profile"
		links 
		{
			"sfml-graphics-s",
			"sfml-window-s",
			"sfml-system-s"
		}

		defines {"RELEASE", "PROFILE"}
		runtime "Release"
		optimize "on"
		symbols "on"
