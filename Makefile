# check code with linters
check:
	poetry run ruff format . --line-length 120 
	poetry run ruff check --line-length 120 --fix


# run streamlit app
run:
	poetry run streamlit run src/app/main.py
