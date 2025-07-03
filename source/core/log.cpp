#include "log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

#ifdef LOG_ON
std::shared_ptr<spdlog::logger> logger::spd_logger;
#endif

void logger::initialize()
{
#ifdef LOG_ON
	spdlog::set_pattern("%^[%T] %n: %v%$");

	spd_logger = spdlog::stdout_color_mt("Log:");
	spd_logger->set_level(spdlog::level::trace);
#endif
}