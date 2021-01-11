-- Display the housing_cleaned_data table data
select * from housing_cleaned_data;

-- Display the city_county_mapping table data
select * from city_county_mapping;

-- Display the covid_cleaned table data
select * from covid_cleaned;

-- Some region_names in housing_cleaned_data table are separated by '-'.
-- This function returns the first city preceded by '-'.
create or replace function first_city(region_name varchar(100))
returns varchar(100)
language plpgsql
as
$$
declare
   first_city_value varchar(100);
begin
   if position('-' in region_name) = 0 then
   		first_city_value = region_name;
	else	
   		SELECT	SUBSTRING (region_name from 1 for (position('-' in region_name)-1)) INTO first_city_value;
	end if;	
	RETURN first_city_value;	
end;
$$;

-- Drop the resultant table if exists
DROP TABLE IF EXISTS fl_ca_housing_data; 

-- Join all the columns from 'housing_cleaned_data' table and the county from 'city_county_mapping' table
-- where city in city_county_mapping table matches with region_name in housing_cleaned_data table.
select h.region_name, h.state_name, c.county_name, h.date, h.avg_price  
INTO fl_ca_housing_data 
FROM housing_cleaned_data as h LEFT OUTER JOIN city_county_mapping as c
ON c.city = first_city(h.region_name);

-- Display the resultant table from the join
select * from fl_ca_housing_data; 

