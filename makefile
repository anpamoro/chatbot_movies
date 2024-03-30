#################### PACKAGE ACTIONS ###################
run_api:
	uvicorn api.api:app --reload
