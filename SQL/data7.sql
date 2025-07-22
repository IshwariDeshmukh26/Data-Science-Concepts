SHOW TIMEZONE

--INSERT INTO timezone VALUES(
--		TIMESTAMP WITHOUT TIME ZONE '2000-01-01 10:00:00-05'
--		TIMESTAMP WITH TIME ZONE '2000-01-01 10:00:00-05'
--	);

CREATE TABLE timezones(
	ts TIMESTAMP WITHOUT TIME ZONE,
	tz TIMESTAMP WITH TIME ZONE
)

INSERT INTO timezones VALUES(
	TIMESTAMP WITHOUT TIME ZONE '2000-01-01 10:00:00-05',
	TIMESTAMP WITH TIME ZONE '2000-01-01 10:00:00-05'
)
SELECT * FROM timezones

SELECT now()::date;
SELECT CURRENT_DATE

SELECT TO_CHAR(CURRENT_DATE,'dd/mm/yy')

SELECT TO_CHAR(CURRENT_DATE,'DDD')

SELECT TO_CHAR(CURRENT_DATE,'WW')

SELECT AGE(date'2003-09-26')

SELECT AGE(date'2003-09-26',date'2024/09/26')AS DAY

SELECT EXTRACT(DAY FROM date'1992-11-13')AS DAY

SELECT EXTRACT(MONTH FROM date'1992-11-13')AS MONTH

SELECT EXTRACT(MONTH FROM date'1992/11/13')AS MONTH

SELECT EXTRACT(YEAR FROM date'1992/11/13')AS YEAR

SELECT DATE_TRUNC('year',date'1992/11/13')

SELECT AGE (birth_date),* FROM employees
WHERE(
	EXTRACT(YEAR FROM AGE(birth_date))
)>60;

SELECT count(emp_no)FROM employees
WHERE EXTRACT(MONTH FROM hire_date)=2

SELECT count(emp_no)FROM employees
WHERE EXTRACT(MONTH FROM birth_date)=11

SELECT MAX(AGE(birth_date)) FROM employees

SELECT MAX(salary) FROM salaries 

--Just modified the query but it will give an error
SELECT *,
	MAX(salary) 
	FROM salaries 


SELECT *,
	MAX(salary) OVER()
	FROM salaries 

SELECT *,
	MAX(salary) OVER()
	FROM salaries
LIMIT 100;


SELECT *,
	MAX(salary) OVER()
	FROM salaries
WHERE salary<70000;

SELECT *,
	AVG(salary) OVER()
	FROM salaries 

SELECT *,
	d.dept_name,
	AVG(salary) OVER()
FROM salaries
JOIN dept_emp AS de USING (emp_no)
JOIN departments AS d USING (dept_no)

SELECT *,
	d.dept_name,
	AVG(salary) OVER(
	PARTITION BY d.dept_name
	)
FROM salaries
JOIN dept_emp AS de USING (emp_no)
JOIN departments AS d USING (dept_no)

SELECT *,
	AVG(salary) OVER(
	PARTITION BY d.dept_name
	)
FROM salaries
JOIN dept_emp AS de USING (emp_no)
JOIN departments AS d USING (dept_no)
