import pickle
p = "/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/new_seeds_seed1/experiment_geo_pbs_rel/latent_geometry.pkl"
d = pickle.load(open(p, "rb"))
names = sorted(d["asymmetry"]["cytokine_names"])
print("\n".join(names))
