import sympy as sp
from sympy import symbols,  simplify, latex, pprint
from typing import List, Dict,  Union
from sympy import Matrix
from 基本再生数 import run_example
#下面都是一些示例模型=======================================================================
def example_sir_model():
    """
    示例1：经典SIR模型
    """
    # 定义符号
    S, I, R = symbols('S I R')
    beta, gamma, N = symbols('beta gamma N', positive=True)
    
    # 定义微分方程组
    equations = {
        'S': -beta * S * I / N,
        'I': beta * S * I / N - gamma * I,
        'R': gamma * I
    }
    
    run_example("示例1：SIR模型", equations, ['I'], {'S': N, 'I': 0, 'R': 0})

def example_seir_model():
    """
    示例2：SEIR模型
    """
    # 定义符号
    S, E, I, R = symbols('S E I R')
    beta, sigma, gamma, N = symbols('beta sigma gamma N', positive=True)
    
    # 定义微分方程组
    equations = {
        'S': -beta * S * I / N,
        'E': beta * S * I / N - sigma * E,
        'I': sigma * E - gamma * I,
        'R': gamma * I
    }
    
    run_example("示例2：SEIR模型", equations, ['E', 'I'], {'S': N, 'E': 0, 'I': 0, 'R': 0})

def example_seir_with_vaccination():
    """
    示例3：带疫苗接种的SEIR模型
    """
    # 定义符号
    S, E, I, R, V = symbols('S E I R V')
    beta, sigma, gamma, nu, N = symbols('beta sigma gamma nu N', positive=True)
    
    # 定义微分方程组
    equations = {
        'S': -beta * S * I / N - nu * S,
        'E': beta * S * I / N - sigma * E,
        'I': sigma * E - gamma * I,
        'R': gamma * I,
        'V': nu * S
    }
    
    run_example("示例3：带疫苗接种的SEIR模型", equations, ['E', 'I'], {'S': N, 'E': 0, 'I': 0, 'R': 0, 'V': 0})

def example_complex_seir_model():
    """
    示例4：复杂的SEIR模型（包含隔离）
    仓室：S, E, E_Q, I, I_Q, I_D, R
    感染相关仓室：E, E_Q, I
    """
    # 定义符号
    S, E, E_Q, I, I_Q, I_D, R = symbols('S E E_Q I I_Q I_D R')
    b, q, u, v, w, N = symbols('b q u v w N', positive=True)
    
    # 定义微分方程组
    equations = {
        'S': -b * S * I / N,
        'E': b * (1 - q) * S * I / N - u * E,
        'E_Q': b * q * S * I / N - u * E_Q,
        'I': u * E - (v + w) * I,
        'I_Q': u * E_Q - (v + w) * I_Q,
        'I_D': w * (I + I_Q) - v * I_D,
        'R': v * (I + I_Q + I_D)
    }

    run_example("示例4：复杂SEIR模型（包含隔离）", equations, ['E', 'E_Q', 'I'])

def example_complex_host_model():
    """
    示例5：宿主模型
    仓室：I V S M
    感染相关仓室：I V
    """
    # 定义符号
    I, V, S, M = symbols('I V S M')
    beta_s, b, gamma, beta_m, c = symbols('beta_s b gamma beta_m c', positive=True)
 
    # 定义微分方程组
    equations = {
        'I': beta_s * S * V - (b + gamma) * I,
        'V': beta_m * M * I - c * V,
        'S': b - b * S + gamma * I - beta_s * S * V,
        'M': c - c * M - beta_m * M * I
    }
    
    run_example("示例5：宿主模型", equations, ['I', 'V'], {'I': 0, 'V': 0, 'S': 1, 'M': 1})

def seiqr_model():
    """
    SEIQR传染病模型（COVID-19专用模型）
    仓室：S E I Q R
    感染相关仓室：E（暴露者）
    特点：包含隔离仓室(Q)，考虑自然死亡和疾病致死
    """
    # 定义符号
    S, E, I, Q, R = symbols('S E I Q R')
    Λ, α, d1, d2, r, β1, β2, σ1, σ2, σ3 = symbols('Λ α d1 d2 r β1 β2 σ1 σ2 σ3', positive=True)
    
    # 组合参数
    ε1 = r + β1 + σ3 + d1
    ε2 = β2 + σ2 + d1 + d2
    ε3 = σ1 + d1 + d2
    
    # 定义微分方程组
    equations = {
        'S': Λ - α * S * E - d1 * S,                 # 易感者：人口流入 - 感染 - 自然死亡
        'E': α * S * E - ε1 * E,                     # 暴露者：感染 - (转化+隔离+康复+死亡)
        'I': r * E - ε2 * I,                         # 感染者：暴露者转化 - (隔离+康复+死亡)
        'Q': β1 * E + β2 * I - ε3 * Q,               # 隔离者：暴露者和感染者转入 - (康复+死亡)
        'R': σ3 * E + σ2 * I + σ1 * Q - d1 * R       # 康复者：各仓室康复 - 自然死亡
    }
    
    run_example("SEIQR传染病模型（COVID-19专用）", equations, ['E'], {'S': Λ/d1, 'E': 0, 'I': 0, 'Q': 0, 'R': 0})

def example_seiqr_model():
    """
    示例：SEIQR模型（隔离模型）
    仓室：S E I Q R
    感染相关仓室：I
    """
    # 定义符号
    S, E, I, Q, R = symbols('S E I Q R')
    beta, sigma, gamma, gamma_q, q, mu = symbols('beta sigma gamma gamma_q q mu', positive=True)
 
    # 定义微分方程组
    equations = {
        'S': mu - beta * S * I - mu * S,                        # 易感者：出生 - 感染 - 自然死亡
        'E': beta * S * I - sigma * E - mu * E,                 # 暴露者：感染 - 发病 - 自然死亡
        'I': sigma * E - gamma * I - q * I - mu * I,            # 感染者：发病 - 康复 - 隔离 - 自然死亡
        'Q': q * I - gamma_q * Q - mu * Q,                      # 隔离者：隔离 - 康复 - 自然死亡
        'R': gamma * I + gamma_q * Q - mu * R                   # 康复者：康复 - 自然死亡
    }
    
    run_example("示例：SEIQR模型（隔离模型）", equations, ['E', 'I'], {'S': 1, 'E': 0, 'I': 0, 'Q': 0, 'R': 0})    

def example_mseir_model():
    """
    示例：MSEIR模型（母体免疫模型）
    仓室：M S E I R
    感染相关仓室：I
    """
    # 定义符号
    M, S, E, I, R = symbols('M S E I R')
    beta, sigma, gamma, delta, mu = symbols('beta sigma gamma delta mu', positive=True)
 
    # 定义微分方程组
    equations = {
        'M': mu - delta * M - mu * M,                   # 母体免疫者：出生 - 免疫力消失 - 自然死亡
        'S': delta * M - beta * S * I - mu * S,         # 易感者：失去母体免疫 - 感染 - 自然死亡
        'E': beta * S * I - sigma * E - mu * E,         # 暴露者：感染 - 发病 - 自然死亡
        'I': sigma * E - gamma * I - mu * I,            # 感染者：发病 - 康复 - 自然死亡
        'R': gamma * I - mu * R                         # 康复者：康复 - 自然死亡
    }
    
    run_example("示例：MSEIR模型（母体免疫模型）", equations, ['E', 'I'], 
                {'M': delta/(delta + mu), 'S': mu/(delta + mu), 'E': 0, 'I': 0, 'R': 0})

def example_seir_model_with_birth_death():
    """
    SEIR模型（带自然出生和死亡）
    仓室：S E I R
    感染相关仓室：I
    特点：考虑人口自然出生和死亡，总人口N恒定
    """
    # 定义符号
    S, E, I, R = symbols('S E I R')
    N, mu, beta, alpha, gamma = symbols('N mu beta alpha gamma', positive=True)
 
    # 定义微分方程组
    equations = {
        'S': mu * N - mu * S - (beta * I * S) / N,  # 易感者：出生 - 自然死亡 - 感染
        'E': (beta * I * S) / N - (mu + alpha) * E,  # 暴露者：感染 - (自然死亡+发病)
        'I': alpha * E - (gamma + mu) * I,           # 感染者：发病 - (康复+自然死亡)
        'R': gamma * I - mu * R                      # 康复者：康复 - 自然死亡
    }
    
    run_example("SEIR模型（带自然出生和死亡）", equations, ['E', 'I'], {'S': N, 'E': 0, 'I': 0, 'R': 0})

def example_seir_model_with_vital_dynamics():
    """
    SEIR模型（带出生和死亡动力学）
    仓室：S E I R
    感染相关仓室：I
    """
    # 定义符号
    S, E, I, R = symbols('S E I R')
    lamda, mu, beta, kappa, gamma = symbols('lambda mu beta kappa gamma', positive=True)
 
    # 定义微分方程组
    equations = {
        'S': lamda - mu * S - beta * S * I,          # 易感者：出生 - 自然死亡 - 感染
        'E': beta * S * I - (mu + kappa) * E,        # 暴露者：感染 - (自然死亡+转为感染者)
        'I': kappa * E - (mu + gamma) * I,           # 感染者：暴露者转化 - (自然死亡+康复)
        'R': gamma * I - mu * R                      # 康复者：康复 - 自然死亡
    }
    run_example("SEIR模型（带出生和死亡动力学）", equations, ['E','I'], {'S': lamda/mu, 'E': 0, 'I': 0, 'R': 0})

from sympy import symbols

def example_msir_model_with_vital_dynamics():
    """
    MSIR模型（带母传免疫和人口动态）
    仓室：M S I R
    感染相关仓室：I
    """
    # 定义符号 (lambda, mu, alpha, beta, gamma)
    M, S, I, R = symbols('M S I R')
    lamda, mu, alpha, beta, gamma = symbols('lambda mu alpha beta gamma', positive=True)

    # 定义微分方程组 (假设 N=1 或 S/N 简化为 S)
    equations = {
        'M': lamda - mu * M - alpha * M,               # 母传免疫者：出生 - 自然死亡 - 免疫衰减
        'S': alpha * M - mu * S - beta * S * I,        # 易感者：免疫衰减获得 - 自然死亡 - 感染
        'I': beta * S * I - (mu + gamma) * I,          # 感染者：感染 - (自然死亡 + 康复)
        'R': gamma * I - mu * R                        # 康复者：康复 - 自然死亡
    }

    # 无病平衡点 (Disease-Free Equilibrium, DFE)
    # M* = lambda / (mu + alpha), S* = alpha * M* / mu, I* = 0, R* = 0
    dfe = {'M': lamda / (mu + alpha), 'S': (alpha * lamda) / (mu * (mu + alpha)), 'I': 0, 'R': 0}
    
    # R0 表达式: R0 = beta / (mu + gamma)
    
    run_example("MSIR模型（带母传免疫和人口动态）", equations, ['I'], dfe)
    #print(f"MSIR 模型的基本再生数 R0 = {beta} / ({mu} + {gamma})")

def example_seirs_model_with_vital_dynamics():
    """
    SEIRS模型（带潜伏期、免疫衰减和人口动态）
    仓室：S E I R
    感染相关仓室：E, I
    """
    # 定义符号 (lambda, mu, beta, sigma, gamma, alpha)
    S, E, I, R = symbols('S E I R')
    lamda, mu, beta, sigma, gamma, alpha = symbols('lambda mu beta sigma gamma alpha', positive=True)

    # 定义微分方程组 (假设 N=1 或 S/N 简化为 S)
    equations = {
        'S': lamda - mu * S - beta * S * I + alpha * R,   # 易感者：出生 - 自然死亡 - 感染 + 免疫衰减
        'E': beta * S * I - (mu + sigma) * E,             # 暴露者：感染 - (自然死亡 + 转为感染者)
        'I': sigma * E - (mu + gamma) * I,                # 感染者：暴露者转化 - (自然死亡 + 康复)
        'R': gamma * I - (mu + alpha) * R                 # 康复者：康复 - (自然死亡 + 免疫衰减)
    }

    # 无病平衡点 (Disease-Free Equilibrium, DFE)
    # S* = lambda / mu, E* = 0, I* = 0, R* = 0
    dfe = {'S': lamda/mu, 'E': 0, 'I': 0, 'R': 0}
    
    # R0 表达式: R0 = (beta * sigma) / ((mu + sigma) * (mu + gamma))
    
    run_example("SEIRS模型（带潜伏期、免疫衰减和人口动态）", equations, ['E','I'], dfe)
    #r0_expression = f"({beta} * {sigma}) / (({mu} + {sigma}) * ({mu} + {gamma}))"
    #print(f"SEIRS 模型的基本再生数 R0 = {r0_expression}")

#主函数
if __name__ == "__main__":
    # 运行所有示例
    example_sir_model()
    example_seir_model()
    example_seir_with_vaccination()
    example_complex_seir_model()
    example_complex_host_model()
    example_seiqr_model()
    example_seiqr_model()
    example_mseir_model()
    example_seir_model_with_birth_death()
    example_seir_model_with_vital_dynamics()
    example_msir_model_with_vital_dynamics()
    example_seirs_model_with_vital_dynamics()
    print("\n\n" + "=" * 80)
    print("计算结束")

    #print("=" * 80)
    #print("""         
    #def 函数名():
    # 定义仓室符号
    #S, E, I, R = symbols('S E I R')
          
    # 定义参数符号
    #lamda, mu, beta, kappa, gamma = symbols('lambda mu beta kappa gamma', positive=True)
          
    # 定义微分方程组
    #equations = {
    #    'S': lamda - mu * S - beta * S * I,          # 易感者：出生 - 自然死亡 - 感染
    #    'E': beta * S * I - (mu + kappa) * E,        # 暴露者：感染 - (自然死亡+转为感染者)
    #    'I': kappa * E - (mu + gamma) * I,           # 感染者：暴露者转化 - (自然死亡+康复)
    #    'R': gamma * I - mu * R                      # 康复者：康复 - 自然死亡
    #}
    
    # 方式1：手动设置无病平衡点 （需要手动设置所有仓室的平衡点）
    #run_example("模型的名称", equations, ['E','I'], {'S': lamda/mu, 'E': 0, 'I': 0, 'R': 0})
    
    # 方式2：自动设置无病平衡点（系统会自动将感染仓室设为0，非感染仓室设为仓室初值的符号例如：S0）
    #run_example("模型的名称", equations, ['E','I'])
    #""")

    #print("\n\n" + "=" * 80)
    #print("""
    #      需要增加模型的话只需要在上面增加一个函数，然后在主函数中调用即可
    #      例如增加一个SEIR模型，只需要增加一个example_seir_model()函数，然后在主函数中调用
    #      例如：example_seir_model()  
    #""")