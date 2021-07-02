def factory_from_config(factory_base, config_data, section_name):
    if section_name == 'None':
        return None
    base_type = config_data.get(section_name, 'type')
    if base_type == 'None':
        return None
    new_object = factory_base(base_type)
    new_object.initialize_from_config(config_data, section_name)
    return new_object