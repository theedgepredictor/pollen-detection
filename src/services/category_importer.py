## pollens
import argparse
import os
import re
import datetime
import string
import pandas as pd

from src.consts import KU_COMMON_NAMES, POLLEN20_NAMES
from src.utils import get_webpage, get_webpage_soup, name_filter, re_braces

WEBSITE_URL = 'https://www.pollenlibrary.com'
def get_all_pollen_types_for_letter(letter:str):
    '''
    Return all sublinks of a navigation page
    '''
    html = get_webpage(f"{WEBSITE_URL}/SEARCH/LETTER/{letter}")
    soup = get_webpage_soup(html)
    a_links = soup.find_all('a')
    if len(a_links) == 0:
        return []

    return [f"{WEBSITE_URL}{a.get('href')}" for a in a_links if '/Specie/' in a.get('href')]

def get_subpage_content(url:str):
    '''
    Scrape the content part of the sub page
    '''
    html = get_webpage(url)
    soup = get_webpage_soup(html)
    content = soup.find('div',{'id':'content'})
    return {
        'url':url,
        'content':str(content),
        'scraped_at':datetime.datetime.now()
    }

class PollenTaxonomyInfo:
    def __init__(
        self,
        friendly_name,
        family,
        genus,
        species,
        latin_name,
        description,
        sub_group_description,
        allergenicity,
        allergy_level,
        pollination_season_text,
        wiki_link=None,
        kingdom = None,
        division = None,
        class_ = None,
        order = None
        ):
        self.friendly_name = friendly_name
        self.family = family
        self.genus = genus
        self.species = species
        self.latin_name = latin_name
        self.description = description
        self.sub_group_description = sub_group_description
        self.allergenicity = allergenicity
        self.allergy_level = allergy_level
        self.pollination_season_text = pollination_season_text
        self.wiki_link = wiki_link
        self.kingdom = kingdom
        self.division = division
        self.class_ = class_
        self.order = order
        self.id = f"{name_filter(self.family)}_{name_filter(self.genus)}_{name_filter(self.species)}"

def parse_subpage_content(content:str):
    '''
    Use BeautifulSoup to pull out content metadata
    '''
    soup = get_webpage_soup(content)
    taxonomy_info = soup.find('div',{'id':'main_lib_content_taxonomy'})
    ### taxonomy sub groups
    family = taxonomy_info.find('span', class_='headerGrn2', string='Family: ').find_next('a').text.lower()
    genus = taxonomy_info.find('span', class_='headerGrn2', string='Genus: ').find_next('a').text
    species_group = taxonomy_info.find('span', class_='headerGrn2', string='Species: ').find_next('i').text
    friendly_name = re_braces(species_group).strip()
    species = re.search(r'\((.*?)\)', species_group).group(1).split(' ')[1]
    latin_name = re.search(r'\((.*?)\)', species_group).group(1).strip()
    try:
        description = soup.find('div',{'id':'main_lib_content_divNativity'}).text
    except Exception as e:
        print('ERR SOUP Description')
        description = None

    try:
        allergenicity = soup.find('div',{'id':'main_lib_content_divAllergenicity'}).text.split(': ')[1]
    except Exception as e:
        print('ERR SOUP allergenicity')
        allergenicity = None
    if allergenicity is not None:
        if 'no allergy' in allergenicity.lower():
            allergy_level = 0
        elif 'mild allergen' in allergenicity.lower():
            allergy_level = 1
        elif 'moderate allergen' in allergenicity.lower():
            allergy_level = 2
        elif 'severe allergen' in allergenicity.lower():
            allergy_level = 3
        else:
            allergy_level = -1
    else:
        allergy_level = -1

    try:
        pollination = soup.find('div',{'id':'main_lib_content_divPollination'}).text.split(': ')[2]
    except Exception as e:
        print('ERR SOUP pollination')
        pollination = None
    try:
        plant_group = soup.find('div',{'id':'main_lib_content_divPlantGroup'}).text
    except Exception as e:
        print('ERR SOUP plant_group')
        plant_group = None

    try:
        alt_links = soup.find('div',{'id':'main_lib_content_genusLinks'})
    except Exception as e:
        print('ERR SOUP alt_links')
        alt_links = None
    if alt_links is not None:
        wiki_link = [a.get('href') for a in alt_links.find_all('a') if 'wikipedia.org' in a.get('href')][0]
    else:
        wiki_link = None
    wiki_data = wiki_subparse(wiki_url=wiki_link)

    obj = PollenTaxonomyInfo(
            friendly_name,
            family = wiki_data['family'] if wiki_data['family'] is not None else family,
            genus = wiki_data['genus'] if wiki_data['genus'] is not None else genus,
            species = wiki_data['species'] if wiki_data['species'] is not None else species,
            latin_name = wiki_data['latin_name'] if wiki_data['latin_name'] is not None else latin_name,
            description=description,
            sub_group_description=plant_group,
            allergenicity=allergenicity,
            allergy_level=allergy_level,
            pollination_season_text = pollination,
            wiki_link=wiki_link,
            kingdom = wiki_data['kingdom'],
            division = wiki_data['division'],
            class_ = wiki_data['class'],
            order = wiki_data['order'],
    )

    return obj.__dict__


def wiki_subparse(wiki_url:str):
    '''
    Determine full taxonomy tree from wiki page
    '''
    taxonomy_tree = {
        'kingdom':None,
        'division':None,
        'class':None,
        'order':None,
        'family':None,
        'genus':None,
        'species':None,
        'latin_name':None
    }
    if wiki_url is None:
        return taxonomy_tree

    html = get_webpage(wiki_url)
    soup = get_webpage_soup(html)
    categories = ['Kingdom', 'Division', 'Class', 'Order', 'Family', 'Genus', 'Species']
    try:
        for category in categories:
            for td in soup.find_all('td'):
                if category in td.text:
                    value = td.find_next_sibling('td').text.strip().replace('.\xa0',' ')
                    taxonomy_tree[category.lower()] = value.split(' ')[1] if category == 'Species' else value
                    break
        taxonomy_tree['latin_name'] =soup.find('span',class_='binomial').find_next('i').text.strip()
    except Exception as e:
        print(e)
    return taxonomy_tree

def raw():
    print("Building Metadata Library for Pollen Identification")
    taxonomy_objs = []
    for letter in string.ascii_uppercase:
        letter_pollen_types = get_all_pollen_types_for_letter(letter)
        for pollen_type_url in letter_pollen_types:
            content_obj = get_subpage_content(pollen_type_url)
            pollen_info_obj = parse_subpage_content(content_obj['content'])
            pollen_info_obj['scrape_url'] = content_obj['url']
            pollen_info_obj['scraped_at'] = content_obj['scraped_at']
            taxonomy_objs.append(pollen_info_obj)
    metadata_df = pd.DataFrame(taxonomy_objs)
    path = './data/metadata/pollenlibrary/'
    os.makedirs(path, exist_ok=True)
    metadata_df.to_json(path+'full.json',orient='records')
    return metadata_df

def preprocessing(metadata_df):

    ## Missing metadata records

    missing = [
        [{'id': 'SALICACEAE_POPULUS_SP', 'friendly_name': 'SALICEAE POPULUS SP'},
         {'id': 'PLATANACEAE_PLATANUS_SP', 'friendly_name': 'PLATANACEA PLATANUS SP'},
         {'id': 'FABACEAE_TRIFOLIUM_SP', 'friendly_name': 'CLOVER'},
         {'id': 'SALICACEAE_SALIX_SP', 'friendly_name': 'WILLOW'},
         {'id': 'APIACEAE_ANGELICA_SP', 'friendly_name': 'ANGELICA'},
         {'id': 'APIACEAE_ANGELICA_ARCHANGELICA', 'friendly_name': 'ANGELICA GARDEN'},
         {'id': 'BUNIAS_ORIENTALIS', 'friendly_name': 'HILL MUSTARD'},
         {'id': 'MALVACEAE_TILIA', 'friendly_name': 'LINDEN'},
         {'id': 'SABATIA_ANGULARIS', 'friendly_name': 'MEADOW PINK'},
         {'id': 'BETULACEAE_ALNUS_SP', 'friendly_name': 'ALDER'},
         {'id': 'BETULACEAE_BETULA_SP', 'friendly_name': 'BIRCH'},
         {'id': 'ONAGRACEAE_CHAMERION_ANGUSTIFOLIUM', 'friendly_name': 'FIREWEED'},
         {'id': 'URTICACEAE_URTICA_SP', 'friendly_name': 'NETTLE'},
         {'id': 'AMARANTHACEAE_AMARANTHUS_SP', 'friendly_name': 'PIGWEED'},
         {'id': 'PLANTAGINACEAE_PLANTAGO_SP', 'friendly_name': 'PLANTAIN'},
         {'id': 'POLYGONACEAE_RUMEX_SP', 'friendly_name': 'SORREL'},
         {'id': 'POACEAE_GE_SP', 'friendly_name': 'GRASS'},
         {'id': 'PINACEAE_PINUS_SP', 'friendly_name': 'PINE'},
         {'id': 'POLYGONACEAE_FAGOPYRUM_ESCULENTUM', 'friendly_name': 'BUCKWHEAT'},
         {'id': 'SAPINDACEAE_ACER_SP', 'friendly_name': 'MAPLE'},
         {'id': 'BETULACEAE_CORYLUS_SP', 'friendly_name': 'HAZEL'}]
    ]

    metadata_df = pd.concat([metadata_df, pd.DataFrame(missing)], ignore_index=True)

    ## Make unique family, genus, species and reformat id column
    metadata_df[['family', 'genus', 'species']] = metadata_df['id'].str.split('_', expand=True)
    metadata_df.family = metadata_df.family.str.upper()
    metadata_df.genus = metadata_df.genus.str.upper()
    metadata_df.species = metadata_df.species.str.upper()
    metadata_df['id'] = metadata_df.family + '_' + metadata_df.genus + '_' + metadata_df.species

    ## Handle active/inactive seasons for each season type based on pollination season text column
    metadata_df['is_active_fall_season'] = 0
    metadata_df['is_active_winter_season'] = 0
    metadata_df['is_active_spring_season'] = 0
    metadata_df['is_active_summer_season'] = 0
    metadata_df.loc[metadata_df.pollination_season_text == 'Summer to Fall.', ['is_active_fall_season', 'is_active_summer_season']] = 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Spring.', 'is_active_spring_season'] = 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Spring to Fall.', ['is_active_spring_season', 'is_active_summer_season', 'is_active_fall_season']] = 1, 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Winter to Summer.', ['is_active_winter_season', 'is_active_spring_season', 'is_active_summer_season']] = 1, 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Spring to Summer.', ['is_active_spring_season', 'is_active_summer_season']] = 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Winter to Spring.', ['is_active_winter_season', 'is_active_spring_season']] = 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Summer.', 'is_active_summer_season'] = 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Summer to Winter.', ['is_active_summer_season', 'is_active_fall_season', 'is_active_winter_season']] = 1, 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Fall.', 'is_active_fall_season'] = 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Spring and Fall.', ['is_active_spring_season', 'is_active_fall_season']] = 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Fall to Winter.', ['is_active_fall_season', 'is_active_winter_season']] = 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Winter.', 'is_active_winter_season'] = 1
    metadata_df.loc[metadata_df.pollination_season_text == 'Fall to Spring.', ['is_active_fall_season', 'is_active_winter_season', 'is_active_spring_season']] = 1, 1, 1
    metadata_df.loc[metadata_df.pollination_season_text == 'all year long.', ['is_active_fall_season', 'is_active_winter_season', 'is_active_spring_season', 'is_active_summer_season']] = 1, 1, 1, 1

    ## Common name table
    latin_name_df = metadata_df[['id', 'latin_name']].rename(columns={'latin_name': 'name'})
    latin_name_df.name = latin_name_df.name.str.upper()
    latin_name_df['classification_level'] = 'SPECIES'

    friendly_name_df = metadata_df[['id', 'friendly_name']].rename(columns={'friendly_name': 'name'})
    friendly_name_df.name = friendly_name_df.name.str.upper()
    friendly_name_df['classification_level'] = 'SPECIES'

    long_latin_name_df = metadata_df[['id']].copy()
    long_latin_name_df['name'] = long_latin_name_df.id.str.replace('_', ' ')
    long_latin_name_df['classification_level'] = 'SPECIES'

    ku_common_names_df = pd.DataFrame(KU_COMMON_NAMES)
    pollen20_common_names_df = pd.DataFrame(POLLEN20_NAMES)

    common_names_df = [
        latin_name_df,
        friendly_name_df,
        long_latin_name_df,
        ku_common_names_df,
        pollen20_common_names_df
    ]

    common_names_df = pd.concat(common_names_df, ignore_index=True)

    return metadata_df, common_names_df


def runner(skip_raw=False):
    '''
    ETL Pipeline for scraping pollen metadata from a website, normalizing naming formats and saving the results to csvs
    :param skip_raw:
    :return:
    '''
    if skip_raw:
        metadata_df = pd.read_json('./data/metadata/pollenlibrary/full.json')
    else:
        metadata_df = raw()

    preprocessed_df, common_names_df = preprocessing(metadata_df)
    path = './data/database/'
    os.makedirs(path, exist_ok=True)
    ### Load preprocessed data as a csv file
    preprocessed_df.to_csv(path+'pollen_categories.csv',index=False)
    common_names_df.to_csv(path + 'pollen_commonnames.csv',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Pollen Metadata Library')
    # If you have already run the raw collection pass True to skip the raw upload
    parser.add_argument('--skip_raw', type=bool, default=False, help='Skip the Raw collection of the metadata library')
    args = parser.parse_args()
    runner(skip_raw=args.skip_raw)