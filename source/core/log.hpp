#pragma once

#ifdef LOG_ON
#pragma warning(disable:4996)
#include <spdlog/spdlog.h>
#pragma warning(default:4996)

#include <spdlog/fmt/ostr.h>
#endif

class logger
{
public:
	static void initialize();

public:
#ifdef LOG_ON
	static std::shared_ptr<spdlog::logger> spd_logger;
#endif
};

#ifdef LOG_ON
#define LOG_ERROR(...) logger::spd_logger->error(__VA_ARGS__)
#define LOG_WARN(...)  logger::spd_logger->warn(__VA_ARGS__)
#define LOG_INFO(...)  logger::spd_logger->info(__VA_ARGS__)
#define LOG_TRACE(...) logger::spd_logger->trace(__VA_ARGS__)
#else
#define LOG_ERROR(...)	
#define LOG_WARN(...)	
#define LOG_INFO(...)	
#define LOG_TRACE(...)	
#endif

#ifdef ASSERT_ON
#define ASSERT(expression, exit_path, ...) if (!(expression)) { LOG_ERROR("Assertion failed: [{}] File [{}] Line [{}]\n{}", #expression, __FILE__, __LINE__, fmt::format(__VA_ARGS__)); __debugbreak(); exit_path; }
#define ASSERT_FATAL(expression, ...) if (!(expression)) { LOG_ERROR("Fatal assertion failed: [{}] File [{}] Line [{}]\n{}", #expression, __FILE__, __LINE__, fmt::format(__VA_ARGS__)); __debugbreak(); abort(); }
#else
#define ASSERT(expression, ...) 
#define ASSERT_FATAL(expression)
#endif