#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

from chembl_webresource_client.new_client import new_client
molecule = new_client.molecule
res = molecule.search('protease')

for i in range(2,5):
    print(i)