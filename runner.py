import os

complexities = [0.0]
experiment_name = '10M'
experiment_type = 'empty'

# complexities = [0.09]
# experiment_name = '10M'
# experiment_type = 'noincrease'

# complexities = [0.01 * i for i in range(10)]
# experiment_name = '10M'
# experiment_type = 'increase'

save_interval = 50000
intervals_per_complexity = 10000000 // save_interval // len(complexities)

last_model = ''
episode = 0
for complexity in complexities:
    for interval in range(intervals_per_complexity):
        episode += 1
        episode_name = f'{experiment_type}_total_{save_interval*intervals_per_complexity*len(complexities)}_timestep_{episode*save_interval}_complexity_{complexity:.02}'

        model_path = f'experiments/{experiment_name}/models/{experiment_type}'
        os.makedirs(model_path, exist_ok=True)
        model_path += f'/{episode_name}.pkl'

        log_path = f'experiments/{experiment_name}/log/{experiment_type}/{episode_name}'
        os.makedirs(log_path, exist_ok=True)

        args = ['python simple_training.py']
        if last_model != '': args.append(f'-i "{last_model}"')
        args.append(f'-o "{model_path}"')
        args.append(f'-l "{log_path}"')
        args.append(f'-t {save_interval}')
        args.append(f'-c {complexity}')
        args.append(f'2> /dev/null')

        print(f'--- Running timesteps {(episode-1)*save_interval}-{episode*save_interval} on complexity {complexity} ({experiment_name}_{experiment_type}) ---')
        print('executing:', ' '.join(args), '\n')
        exit_code = os.system(' '.join(args))
        if exit_code != 0:
            print(f'--- Exit code was not 0, was {exit_code} ---')
            exit(1)

        last_model = model_path
