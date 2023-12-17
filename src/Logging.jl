using Logging

# an exemplary logging
# 17/12/23

a = [1,2,3.] # data

global_logger(ConsoleLogger(LogLevel(3))) # How to set Log_Level
# global_logger(ConsoleLogger(Info))
# global_logger(ConsoleLogger(Debug))
# global_logger(ConsoleLogger(Warn))
# global_logger(ConsoleLogger(Error))

@debug "m_debug" -1000a = -1000a
@info "m_info" 0a = 0a

@logmsg LogLevel(1) "m_1" 1a = 1a
@logmsg LogLevel(2) "m_2" 2a = 2a
@logmsg LogLevel(3) "m_3" 3a = 3a
@logmsg LogLevel(4) "m_4" 4a = 4a

@warn "m_warn" 1000a = 1000a
@error "m_error" 2000a = 2000a
