import { API_ADDR } from './consts';

export function doForm() {
    const results = document.querySelector('.results');

    document.querySelector('#form').addEventListener('submit', (e) => {
        e.preventDefault();

        results.classList.remove('show');
        results.querySelector('.results--additionalmsg').classList.add('hidden');

        const formProps = Object.fromEntries(new FormData(e.target));
        const passedData = {};

        Object.keys(formProps).forEach((d) => {
            if (formProps[d] === 'yes') passedData[d] = true;
            else if (formProps[d] === 'no') passedData[d] = false;
            else {
                const tmp = parseInt(formProps[d]);

                if (!isNaN(tmp) && d !== 'age_category') {
                    passedData[d] = tmp;
                } else {
                    passedData[d] = formProps[d];
                }
            }
        });

        const weight = passedData['weight'];
        delete passedData['weight'];
        const height = passedData['height'];
        delete passedData['height'];

        const bmi = weight / ((height * height) / 10000);

        passedData['bmi'] = bmi;

        fetch(`${API_ADDR}/classify`, {
            method: 'post',
            body: JSON.stringify(passedData),
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then((resp) => resp.json())
            .then((json) => {
                let yes = parseFloat(json['yes']);

                results.querySelector('#results-percentage').innerHTML = `${(yes * 100).toFixed(
                    0
                )}%`;

                if (yes >= 0.6) {
                    results.querySelector('.results--additionalmsg').classList.remove('hidden');
                }

                results.classList.add('show');
            })
            .catch((err) => console.error(err));
    });
}
