CREATE DATABASE covid_property_pandemic
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

CREATE TABLE cleaned_house_data_CA(
	city VARCHAR(30) NOT NULL,
	month_year VARCHAR(30) NOT NULL,
	median_house_price BIGINT NOT NULL,
	PRIMARY KEY (city)
);

CREATE TABLE cleaned_house_data_FL(
	city VARCHAR(30) NOT NULL,
	month_year VARCHAR(30) NOT NULL,
	median_house_price BIGINT NOT NULL,
	PRIMARY KEY (city)
);

CREATE TABLE cleaned_covid_data_CA(
	city VARCHAR(30) NOT NULL,
	month_year VARCHAR(30) NOT NULL,
	number_of_cases BIGINT NOT NULL,
	PRIMARY KEY (city)
);

CREATE TABLE cleaned_covid_data_FL(
	city VARCHAR(30) NOT NULL,
	month_year VARCHAR(30) NOT NULL,
	number_of_cases BIGINT NOT NULL,
	PRIMARY KEY (city)
);