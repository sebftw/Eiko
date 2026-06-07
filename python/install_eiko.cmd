#!/bin/bash
# ==============================================================================
# LINUX RUNNER (Bash)
# ==============================================================================
REMOTE_INSTALLER_URL="https://raw.githubusercontent.com/sebftw/Eiko/main/python/install_eiko_ubuntu.sh"

# 1. Check if eiko is already available globally
if command -v python3 >/dev/null 2>&1 && python3 -c "import eiko.eiko_torch" >/dev/null 2>&1; then
    echo -e "\033[1;32m-> Eiko is available globally. No virtual environment needed.\033[0m"
    exit 0
else
	# 2. Fallback to dedicated Eiko sandbox
	VENV_PATH="$HOME/eiko"

	# Check if venv exists and eiko module is installed inside it
	if [ ! -f "$VENV_PATH/bin/activate" ] || ! "$VENV_PATH/bin/python" -c "import eiko.eiko_torch" >/dev/null 2>&1; then
		echo -e "\033[1;33m-> Eiko not found. Downloading and launching installer...\033[0m"
		
		TEMP_SH=$(mktemp)
		if curl -sSf "$REMOTE_INSTALLER_URL" -o "$TEMP_SH"; then
			bash "$TEMP_SH"
			SH_EXIT=$?
			rm -f "$TEMP_SH"
			if [ $SH_EXIT -ne 0 ]; then
				echo -e "\033[0;31m-> Installation failed. Exiting.\033[0m"
				exit 1
			fi
		else
			echo -e "\033[0;31m-> Failed to download the Linux installer script. Check connection.\033[0m"
			rm -f "$TEMP_SH"
			exit 1
		fi
	fi

	# 3. Activate the environment
	echo -e "\033[1;32m-> Activating Eiko virtual environment...\033[0m"

	# The magic trick: We source it first, then use 'exec bash --rcfile' 
	# to keep the modifications alive in a fresh interactive shell.
	source "$VENV_PATH/bin/activate"
fi