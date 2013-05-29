log_file = 'pygfit.log'
warn_file = 'pygfit.warn'
logging_off = False
log_open = False
log_fp = ''
warn_fp = ''

def start( new_log='', new_warn='' ):
	global logging_off,log_fp,warn_fp,log_open,log_file,warn_file

	if new_log != '': log_file = new_log
	if new_warn != '': warn_file = new_warn

	if logging_off: return True

	log_fp = open( log_file, 'wb' )
	warn_fp = open( warn_file, 'wb' )
	log_open = True

def log( to_log ):
	global logging_off,log_open,log_fp

	if logging_off: return True
	if not log_open: start()

	if to_log[-1] != "\n": to_log += "\n"
	log_fp.write( to_log )

def warn( to_warn, silent=False ):
	global logging_off,log_open,warn_fp

	if logging_off: return True
	if not log_open: start()

	if to_warn[-1] != "\n": to_warn += "\n"
	if not silent: print to_warn[:-1]
	warn_fp.write( to_warn )

def disable_logging():
	global logging_off

	close()
	logging_off = True

def enable_logging():
	global logging_off

	logging_off = False

def close():
	global log_open,log_fp,warn_fp

	if not log_open: return True
	log_fp.close()
	warn_fp.close()
	log_open = False