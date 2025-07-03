workspace "cart-pole-rl"
	architecture "x64"
	startproject "cart-pole-rl"
	
	configurations
	{
		"debug",
		"assert",
		"release",
		"profile",
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "thirdparty/ImGui"
include "thirdparty/GLFW"
include "thirdparty/glad"

project "cart-pole-rl"
	location ""
	kind "ConsoleApp"
	language "c++"
	cppdialect "c++20"

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
		"thirdparty/Eigen",
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
		
		defines {"DEBUG", "LOG_ON", "ASSERT_ON"}
		runtime "Debug"
		symbols "on"
	
	filter "configurations:assert"
		links 
		{
			"sfml-graphics-s",
			"sfml-window-s",
			"sfml-system-s"
		}

		defines {"RELEASE", "LOG_ON", "ASSERT_ON"}
		runtime "Release"
		optimize "on"

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
