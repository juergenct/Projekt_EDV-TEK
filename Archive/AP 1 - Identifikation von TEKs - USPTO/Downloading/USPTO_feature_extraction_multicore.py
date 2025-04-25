import os
import xml.etree.ElementTree as ET
import pandas as pd
import xmltodict
import multiprocessing as mp


# Create function that processes xml file
def process_xml_file(file):
    
    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23/2023"    

    # Initialize a Pandas dataframe to store data
    result_patent_data_df = pd.DataFrame()

    # Create a list for faulty xml files
    faulty_xml_files = []

    # Read in xml document as string, include folder path
    with open(os.path.join(folder_path, file)) as in_fh:
    # with open(file) as in_fh:
        xml_string = in_fh.read()
    # Split the string at the xml declaration '<?xml'
    xml_list = xml_string.split('<?xml')
    # Remove the first element of the list, which is empty
    xml_list.pop(0)
    # Add the xml declaration to each xml string
    xml_list = ['<?xml' + xml for xml in xml_list]

    # Initialize a dictionary to store the cleaned data
    cleaned_data = {}

    # Process xml strings from xml string list
    for xml in xml_list:        
        if xml:
            try: 
                # Put the xml string into a dictionary
                data = xmltodict.parse(xml)
            except:
                faulty_xml_files.append(file)
                continue 

            # Check whether us-patent-grant is in the dictionary
            if 'us-patent-grant' not in data:
                continue

            # Include document number
            cleaned_data['doc_number'] = data['us-patent-grant']['us-bibliographic-data-grant']['publication-reference']['document-id']['doc-number']
            cleaned_data['date'] = data['us-patent-grant']['us-bibliographic-data-grant']['publication-reference']['document-id']['date']

            # # Include IPC classification
            # if 'classifications-ipcr' in data['us-patent-grant']['us-bibliographic-data-grant']:
            #     cleaned_data['ipc'] = data['us-patent-grant']['us-bibliographic-data-grant']['classifications-ipcr']
            # else:
            #     cleaned_data['ipc'] = {}

            # Include CPC classification
            if 'classifications-cpc' in data['us-patent-grant']['us-bibliographic-data-grant']:
                cleaned_data['main_cpc'] = data['us-patent-grant']['us-bibliographic-data-grant']['classifications-cpc']['main-cpc']
            else:
                cleaned_data['main_cpc'] = {}
            
            # Include further CPC classifications
            if 'classifications-cpc' in data['us-patent-grant']['us-bibliographic-data-grant']:
                if 'further-cpc' in data['us-patent-grant']['us-bibliographic-data-grant']['classifications-cpc']:
                    cleaned_data['further_cpcs'] = data['us-patent-grant']['us-bibliographic-data-grant']['classifications-cpc']['further-cpc']
            else:
                cleaned_data['further_cpcs'] = {}

            # Include title
            if 'invention-title' in data['us-patent-grant']['us-bibliographic-data-grant']:
                if '#text' in data['us-patent-grant']['us-bibliographic-data-grant']['invention-title']:
                    cleaned_data['title'] = data['us-patent-grant']['us-bibliographic-data-grant']['invention-title']['#text']
            else:
                cleaned_data['title'] = ''
            

            # Inlcude applicants information
            # cleaned_data['applicants'] = data['us-patent-grant']['us-bibliographic-data-grant'].get('us-parties', {}).get('us-applicants', {})

            # Include inventor information
            # cleaned_data['inventors'] = data['us-patent-grant']['us-bibliographic-data-grant'].get('us-parties', {}).get('inventors', {})

            # Include assignee information
            # cleaned_data['assignees'] = data['us-patent-grant']['us-bibliographic-data-grant'].get('assignees', {}).get('assignee', {})

            # Include abstract
            if 'abstract' in data['us-patent-grant']:
                if 'p' in data['us-patent-grant']['abstract']:
                    if '#text' in data['us-patent-grant']['abstract']['p']:
                        cleaned_data['abstract'] = data['us-patent-grant']['abstract']['p']['#text']
            else:
                cleaned_data['abstract'] = ''

            # Include claims - Gives 2 Rows instead of 1
            # cleaned_data['claims'] = data['us-patent-grant']['claims']

            # Include us-references-cited
            # cleaned_data['us_cit'] = data['us-patent-grant']['us-bibliographic-data-grant'].get('us-references-cited', {})

            # Include us-related-documents as one string] - Gives 3 Rows instead of 1
            # cleaned_data['us_related_doc'] = data['us-patent-grant']['us-bibliographic-data-grant']['us-related-documents']

    # Append cleaned data to the Pandas DataFrame
    cleaned_data_df = pd.DataFrame(cleaned_data)
    result_patent_data_df = pd.concat([result_patent_data_df, cleaned_data_df], ignore_index=True)

    # Write the DataFrame to a json file taking the document number as index
    result_patent_data_df.to_json(os.path.join('/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23 Preprocessed/JSON', file, '2023.json'), orient='records')
    
    return

if __name__ == '__main__':

    # Define the folder path
    folder_path = "/Users/juergenthiesen/Documents/Full Text Bulk Data USPTO 02_23/2023"

    # Get a list of all USPTO files in the folder
    files = os.listdir(folder_path)

    # Filter the list to include only files with the ".xml" extension
    xml_files = [f for f in files if f.endswith('.xml')]

    # Define multiprocessing pool
    num_CPUs = mp.cpu_count() - 4
    pool = mp.Pool(processes=num_CPUs)
    pool.map(process_xml_file, xml_files)
    pool.close()



    