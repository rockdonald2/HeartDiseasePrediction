import { API_ADDR } from "./consts";

export async function doPopulate() {
    const internalPopulate = (where, from) => {
        fetch(from)
            .then(resp => resp.json())
            .then(json => {
                where.innerHTML = json[`${Object.keys(json)[0]}`].map(d => `<option value="${d}">${d}</option>`).join('');
            })
            .catch(err => console.error(err));
    }

    internalPopulate(document.querySelector('#data-gen_health'), `${API_ADDR}/get/gen_health`);
    internalPopulate(document.querySelector('#data-race'), `${API_ADDR}/get/race`);
    internalPopulate(document.querySelector('#data-age_category'), `${API_ADDR}/get/age`);
    internalPopulate(document.querySelector('#data-sex'), `${API_ADDR}/get/sex`);
}